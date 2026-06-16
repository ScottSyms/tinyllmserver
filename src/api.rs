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

use crate::embeddings::Embedder;
use crate::llm::{GenRequest, LlmHandle};

#[derive(Clone)]
pub struct AppState {
    pub llm: LlmHandle,
    pub embedder: Arc<Embedder>,
    pub model_id: String,
    pub embed_id: String,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state)
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
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
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
    content: String,
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

/// Render messages using Gemma's chat template. Gemma has no system role, so a
/// system message is prepended to the first user turn. BOS is added at tokenize.
fn build_gemma_prompt(messages: &[ChatMessage]) -> String {
    let mut system = String::new();
    let mut prompt = String::new();
    let mut first_user_done = false;

    for m in messages {
        match m.role.as_str() {
            "system" => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&m.content);
            }
            "assistant" | "model" => {
                prompt.push_str("<start_of_turn>model\n");
                prompt.push_str(&m.content);
                prompt.push_str("<end_of_turn>\n");
            }
            // treat "user" and anything else as a user turn
            _ => {
                prompt.push_str("<start_of_turn>user\n");
                if !first_user_done && !system.is_empty() {
                    prompt.push_str(&system);
                    prompt.push_str("\n\n");
                    first_user_done = true;
                }
                prompt.push_str(&m.content);
                prompt.push_str("<end_of_turn>\n");
            }
        }
    }

    prompt.push_str("<start_of_turn>model\n");
    prompt
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

    let gen = GenRequest {
        prompt: build_gemma_prompt(&req.messages),
        max_tokens: req.max_tokens.unwrap_or(512).clamp(1, 8192),
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

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", now()),
        object: "chat.completion",
        created: now(),
        model: s.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant".into(),
                content: out.text,
            },
            finish_reason: out.finish_reason.into(),
        }],
        usage: Usage {
            prompt_tokens: out.prompt_tokens,
            completion_tokens: out.completion_tokens,
            total_tokens: out.prompt_tokens + out.completion_tokens,
        },
    }))
}

// ---------- /v1/embeddings ----------

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    input: EmbeddingInput,
}

#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
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

    let embedder = s.embedder.clone();
    let vectors = tokio::task::spawn_blocking(move || embedder.embed(texts))
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding",
            index,
            embedding,
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
