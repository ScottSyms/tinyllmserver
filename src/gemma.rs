//! Gemma 4 chat templating and tool-call (de)serialization.
//!
//! We render the model's own Jinja chat template (extracted from the GGUF) with
//! minijinja, passing the OpenAI `messages` and `tools` straight through. That
//! gives the exact prompt the model was trained on — including Gemma 4's tool
//! definition blocks (`<|tool>…<tool|>`) and turn tokens (`<|turn>…<turn|>`).
//!
//! On the way back, the model emits tool calls in Gemma's notation
//! (`<|tool_call>call:NAME{arg:<|"|>val<|"|>}<tool_call|>`), which we parse into
//! standard OpenAI `tool_calls` with JSON arguments.

use anyhow::{anyhow, Result};
use minijinja::{context, Environment, Value as JValue};
use serde_json::{Map, Value};

/// Gemma's string delimiter token, used around every string literal.
const STR_DELIM: &str = "<|\"|>";

pub fn build_env(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();
    // Enable Python-style methods the template relies on (.get, .strip, …).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template_owned("chat", template)
        .map_err(|e| anyhow!("failed to parse chat template: {e}"))?;
    Ok(env)
}

/// Render the prompt from OpenAI-style messages and tools.
pub fn render_prompt(
    env: &Environment,
    mut messages: Vec<Value>,
    tools: Option<Value>,
) -> Result<String> {
    // OpenAI sends assistant tool-call arguments as a JSON *string*; the template
    // serializes a mapping into Gemma notation but echoes a string verbatim, so
    // parse it into an object first to keep history in the native format.
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
            bos_token => "<bos>",
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
    // Drop any reasoning/thinking channel; it is not part of the reply.
    let stripped = remove_spans(raw, "<|channel>", "<channel|>");

    let mut tool_calls = Vec::new();
    let mut content = String::new();
    let mut rest = stripped.as_str();

    while let Some(start) = rest.find("<|tool_call>") {
        content.push_str(&rest[..start]);
        let after = &rest[start + "<|tool_call>".len()..];
        match after.find("<tool_call|>") {
            Some(end) => {
                if let Some(tc) = parse_call(&after[..end]) {
                    tool_calls.push(tc);
                }
                rest = &after[end + "<tool_call|>".len()..];
            }
            None => {
                if let Some(tc) = parse_call(after) {
                    tool_calls.push(tc);
                }
                rest = "";
                break;
            }
        }
    }
    content.push_str(rest);

    let content = content.trim();
    let content = if content.is_empty() {
        None
    } else {
        Some(content.to_string())
    };
    Parsed { content, tool_calls }
}

/// Parse `call:NAME{ …gemma notation… }` into a name + JSON-string arguments.
fn parse_call(body: &str) -> Option<ParsedToolCall> {
    let body = body.trim().strip_prefix("call:").unwrap_or(body.trim());
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
