//! Model-agnostic chat templating and tool-call (de)serialization.
//!
//! Prompts are rendered with the model's own Jinja chat template (extracted from
//! the GGUF) via minijinja, passing the OpenAI `messages` and `tools` straight
//! through. That makes the prompt side work for any model whose GGUF ships a
//! template (Gemma, Qwen, …).
//!
//! Parsing the *output* is model-specific, so we recognize the two common
//! shapes:
//!   * Qwen / Hermes:  `<tool_call>{"name":…,"arguments":{…}}</tool_call>`
//!   * Gemma 4:        `<|tool_call>call:NAME{arg:<|"|>val<|"|>}<tool_call|>`
//! and strip both thinking conventions (`<think>…</think>`, `<|channel>…`).

use anyhow::{anyhow, Result};
use minijinja::{context, Environment, Value as JValue};
use serde_json::{Map, Value};

/// Gemma's string delimiter token, used around every string literal.
const STR_DELIM: &str = "<|\"|>";

pub fn build_env(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();
    // Enable Python-style methods the templates rely on (.get, .strip, .split…).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template_owned("chat", template)
        .map_err(|e| anyhow!("failed to parse chat template: {e}"))?;
    Ok(env)
}

/// Render the prompt from OpenAI-style messages and tools. `bos_token` is the
/// model's actual BOS string (empty if it has none); templates that prepend BOS
/// reference it, others ignore it.
pub fn render_prompt(
    env: &Environment,
    mut messages: Vec<Value>,
    tools: Option<Value>,
    bos_token: &str,
) -> Result<String> {
    // OpenAI sends assistant tool-call arguments as a JSON *string*; some
    // templates serialize a mapping differently than a raw string, so parse it
    // into an object first to keep history in the native format.
    for m in &mut messages {
        if m.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(tcs) = m.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for tc in tcs {
                    if let Some(func) = tc.get_mut("function") {
                        if let Some(s) = func.get("arguments").and_then(Value::as_str) {
                            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                                func["arguments"] = parsed;
                            }
                        }
                    }
                }
            }
        }
    }

    let tmpl = env.get_template("chat")?;
    let rendered = tmpl
        .render(context! {
            messages => JValue::from_serialize(&messages),
            tools => JValue::from_serialize(&tools),
            add_generation_prompt => true,
            bos_token => bos_token,
        })
        .map_err(|e| anyhow!("template render failed: {e}"))?;
    Ok(rendered)
}

pub struct ParsedToolCall {
    pub name: String,
    /// Arguments as a JSON object string (OpenAI's `function.arguments`).
    pub arguments: String,
}

pub struct Parsed {
    pub content: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

/// Split a raw model turn into free-text content and structured tool calls.
pub fn parse_completion(raw: &str) -> Parsed {
    // Drop reasoning/thinking; it is not part of the reply.
    let s = remove_spans(raw, "<|channel>", "<channel|>");
    let mut s = remove_spans(&s, "<think>", "</think>");
    // Qwen opens <think> in the prompt, so the output may carry only the closer.
    if let Some(idx) = s.rfind("</think>") {
        s = s[idx + "</think>".len()..].to_string();
    }

    // Pick the format by which marker the model used.
    let (content, tool_calls) = if s.contains("<tool_call>") {
        extract(&s, "<tool_call>", "</tool_call>", parse_qwen_call)
    } else {
        extract(&s, "<|tool_call>", "<tool_call|>", parse_gemma_call)
    };

    let content = content.trim();
    let content = if content.is_empty() {
        None
    } else {
        Some(content.to_string())
    };
    Parsed { content, tool_calls }
}

/// Pull every `open … close` block out of `s`, parsing each with `parse`; the
/// text outside the blocks is returned as content.
fn extract<F>(s: &str, open: &str, close: &str, parse: F) -> (String, Vec<ParsedToolCall>)
where
    F: Fn(&str) -> Option<ParsedToolCall>,
{
    let mut calls = Vec::new();
    let mut content = String::new();
    let mut rest = s;
    while let Some(i) = rest.find(open) {
        content.push_str(&rest[..i]);
        let after = &rest[i + open.len()..];
        match after.find(close) {
            Some(j) => {
                if let Some(tc) = parse(&after[..j]) {
                    calls.push(tc);
                }
                rest = &after[j + close.len()..];
            }
            None => {
                if let Some(tc) = parse(after) {
                    calls.push(tc);
                }
                rest = "";
                break;
            }
        }
    }
    content.push_str(rest);
    (content, calls)
}

/// Qwen/Hermes: the block body is a JSON object `{"name":…,"arguments":…}`.
fn parse_qwen_call(inner: &str) -> Option<ParsedToolCall> {
    let v: Value = serde_json::from_str(inner.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let args = v
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    let arguments = match args {
        Value::String(s) => s,
        other => serde_json::to_string(&other).ok()?,
    };
    Some(ParsedToolCall { name, arguments })
}

/// Gemma 4: the block body is `call:NAME{ …gemma notation… }`.
fn parse_gemma_call(inner: &str) -> Option<ParsedToolCall> {
    let body = inner.trim().strip_prefix("call:").unwrap_or(inner.trim());
    let brace = body.find('{')?;
    let name = body[..brace].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut p = Parser { s: body, i: brace };
    let value = p.parse_value()?;
    let arguments = serde_json::to_string(&value).ok()?;
    Some(ParsedToolCall { name, arguments })
}

/// Remove every `open … close` span (inclusive) from `s`.
fn remove_spans(s: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(open) {
        out.push_str(&rest[..i]);
        let after = &rest[i + open.len()..];
        match after.find(close) {
            Some(j) => rest = &after[j + close.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Recursive-descent parser for Gemma's value notation.
struct Parser<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.i..]
    }

    fn skip_ws(&mut self) {
        while self.i < self.s.len() && self.s.as_bytes()[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn parse_value(&mut self) -> Option<Value> {
        self.skip_ws();
        let r = self.rest();
        if let Some(after) = r.strip_prefix(STR_DELIM) {
            self.i += STR_DELIM.len();
            let end = after.find(STR_DELIM)?;
            let val = after[..end].to_string();
            self.i += end + STR_DELIM.len();
            return Some(Value::String(val));
        }
        match self.s.as_bytes().get(self.i)? {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            _ => Some(self.parse_bare()),
        }
    }

    fn parse_object(&mut self) -> Option<Value> {
        self.i += 1; // consume '{'
        let mut map = Map::new();
        loop {
            self.skip_ws();
            if self.rest().starts_with('}') {
                self.i += 1;
                break;
            }
            let colon = self.rest().find(':')?;
            let key = self.rest()[..colon].trim().to_string();
            self.i += colon + 1;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            match self.rest().as_bytes().first() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => break,
            }
        }
        Some(Value::Object(map))
    }

    fn parse_array(&mut self) -> Option<Value> {
        self.i += 1; // consume '['
        let mut arr = Vec::new();
        loop {
            self.skip_ws();
            if self.rest().starts_with(']') {
                self.i += 1;
                break;
            }
            arr.push(self.parse_value()?);
            self.skip_ws();
            match self.rest().as_bytes().first() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => break,
            }
        }
        Some(Value::Array(arr))
    }

    fn parse_bare(&mut self) -> Value {
        let r = self.rest();
        let end = r.find([',', '}', ']']).unwrap_or(r.len());
        let tok = r[..end].trim().to_string();
        self.i += end;
        match tok.as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" => Value::Null,
            _ => {
                if let Ok(n) = tok.parse::<i64>() {
                    Value::Number(n.into())
                } else if let Ok(f) = tok.parse::<f64>() {
                    serde_json::Number::from_f64(f)
                        .map(Value::Number)
                        .unwrap_or(Value::String(tok))
                } else {
                    Value::String(tok)
                }
            }
        }
    }
}
