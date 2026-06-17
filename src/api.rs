use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::chat;
use crate::embeddings::Embedder;
use crate::llm::{GenRequest, LlmHandle};

#[derive(Clone)]
pub struct AppState {
    pub llm: LlmHandle,
    pub embedder: Arc<Embedder>,
    pub env: Arc<minijinja::Environment<'static>>,
    pub bos_token: Arc<str>,
    pub default_max_tokens: usize,
    pub model_id: String,
    pub embed_id: String,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        // Allow browser-based clients (extensions, web apps) to call the API
        // cross-origin. Safe here because the server binds to localhost only.
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state)
}

/// Human-friendly landing page so hitting the server in a browser shows
/// something useful instead of a bare 404. The actual API is JSON over POST.
async fn index(State(s): State<AppState>) -> Response {
    let html = format!(
        r#"<!doctype html><html><head><meta charset="utf-8">
<title>tinymodelserver</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem;color:#222}}
code{{background:#f3f3f3;padding:.1em .3em;border-radius:4px}}
.m{{color:#0a7}}.p{{color:#a60}}</style></head><body>
<h1>tinymodelserver</h1>
<p>OpenAI-compatible local server. This is a JSON API — point a client at
<code>{base}/v1</code>.</p>
<ul>
<li><span class="m">GET</span> <a href="/health">/health</a></li>
<li><span class="m">GET</span> <a href="/v1/models">/v1/models</a></li>
<li><span class="p">POST</span> <code>/v1/chat/completions</code> &nbsp;(model: <code>{chat}</code>)</li>
<li><span class="p">POST</span> <code>/v1/embeddings</code> &nbsp;(model: <code>{embed}</code>)</li>
</ul>
<p>The POST endpoints can't be opened directly in a browser. Try:</p>
<pre>curl {base}/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{{"model":"{chat}","messages":[{{"role":"user","content":"Hello"}}]}}'</pre>
</body></html>"#,
        base = "http://127.0.0.1:8080",
        chat = s.model_id,
        embed = s.embed_id,
    );
    axum::response::Html(html).into_response()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Uniform error response in OpenAI's error envelope shape.
struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.0,
            Json(json!({ "error": { "message": self.1, "type": "server_error" } })),
        )
            .into_response()
    }
}

// ---------- /v1/models ----------

async fn list_models(State(s): State<AppState>) -> impl IntoResponse {
    let created = now();
    Json(json!({
        "object": "list",
        "data": [
            { "id": s.model_id, "object": "model", "created": created, "owned_by": "tinymodelserver" },
            { "id": s.embed_id, "object": "model", "created": created, "owned_by": "tinymodelserver" },
        ]
    }))
}

// ---------- /v1/chat/completions ----------

#[derive(Deserialize)]
struct ChatRequest {
    /// Raw OpenAI messages, passed straight to the model's chat template
    /// (handles system/user/assistant/tool roles, tool_calls, content parts).
    #[serde(default)]
    messages: Vec<serde_json::Value>,
    /// OpenAI tool definitions, forwarded to the template's tool block.
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    seed: Option<u32>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatResponseMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}

#[derive(Serialize)]
struct ToolCallOut {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionOut,
}

#[derive(Serialize)]
struct FunctionOut {
    name: String,
    /// JSON-encoded argument object, per the OpenAI spec.
    arguments: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

async fn chat_completions(
    State(s): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "`messages` must not be empty".into(),
        ));
    }

    let n_tools = req
        .tools
        .as_ref()
        .and_then(|t| t.as_array())
        .map_or(0, |a| a.len());
    tracing::debug!(
        "chat request: {} message(s), {} tool(s)",
        req.messages.len(),
        n_tools
    );

    let prompt = chat::render_prompt(&s.env, req.messages.clone(), req.tools.clone(), &s.bos_token)
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, format!("prompt rendering failed: {e}")))?;
    tracing::debug!("rendered prompt:\n{prompt}");

    let gen = GenRequest {
        prompt,
        add_bos: false, // the template emits <bos> itself
        max_tokens: req.max_tokens.unwrap_or(s.default_max_tokens).clamp(1, 32768),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.95),
        top_k: req.top_k.unwrap_or(40),
        seed: req.seed.unwrap_or_else(|| now() as u32),
    };

    let out = s
        .llm
        .generate(gen)
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    tracing::debug!("raw completion:\n{}", out.text);
    let parsed = chat::parse_completion(&out.text);
    tracing::debug!(
        "parsed: {} tool call(s), content {} chars",
        parsed.tool_calls.len(),
        parsed.content.as_deref().map_or(0, str::len)
    );

    let (content, tool_calls, finish_reason) = if parsed.tool_calls.is_empty() {
        (parsed.content, None, out.finish_reason.to_string())
    } else {
        let calls = parsed
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(i, tc)| ToolCallOut {
                id: format!("call_{}_{i}", now()),
                kind: "function",
                function: FunctionOut {
                    name: tc.name,
                    arguments: tc.arguments,
                },
            })
            .collect();
        (parsed.content, Some(calls), "tool_calls".to_string())
    };

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", now()),
        object: "chat.completion",
        created: now(),
        model: s.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant".into(),
                content,
                tool_calls,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: out.prompt_tokens,
            completion_tokens: out.completion_tokens,
            total_tokens: out.prompt_tokens + out.completion_tokens,
        },
    }))
}

// ---------- /v1/embeddings ----------

/// Encode a float vector as base64 of its raw little-endian f32 bytes — the
/// layout the OpenAI SDKs expect when decoding `encoding_format: "base64"`.
fn encode_f32_base64(v: &[f32]) -> String {
    use base64::Engine;
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    input: EmbeddingInput,
    /// "float" (default) or "base64". The official OpenAI SDKs default to
    /// "base64" and decode it client-side, so we must honor it.
    #[serde(default)]
    encoding_format: Option<String>,
}

/// An embedding is serialized either as a JSON float array or, for
/// `encoding_format: "base64"`, as a base64 string of little-endian f32 bytes.
#[derive(Serialize)]
#[serde(untagged)]
enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    index: usize,
    embedding: EmbeddingValue,
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
    usage: EmbeddingUsage,
}

#[derive(Serialize)]
struct EmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

async fn embeddings(
    State(s): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let texts = match req.input {
        EmbeddingInput::Single(t) => vec![t],
        EmbeddingInput::Many(v) => v,
    };
    if texts.is_empty() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "`input` must not be empty".into(),
        ));
    }
    let approx_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();

    let as_base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                format!("unsupported encoding_format: {other}"),
            ))
        }
    };

    let embedder = s.embedder.clone();
    let vectors = tokio::task::spawn_blocking(move || embedder.embed(texts))
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, v)| EmbeddingData {
            object: "embedding",
            index,
            embedding: if as_base64 {
                EmbeddingValue::Base64(encode_f32_base64(&v))
            } else {
                EmbeddingValue::Float(v)
            },
        })
        .collect();

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model: s.embed_id.clone(),
        usage: EmbeddingUsage {
            prompt_tokens: approx_tokens,
            total_tokens: approx_tokens,
        },
    }))
}
