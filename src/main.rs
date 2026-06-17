mod api;
mod chat;
mod config;
mod embeddings;
mod llm;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

use api::AppState;
use config::Config;
use embeddings::Embedder;
use llm::LlmConfig;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tinymodelserver=info,warn".into()),
        )
        .init();

    let cfg = Config::parse();

    let backend = if cfg!(target_os = "macos") {
        "OxiBonsai (Metal)"
    } else {
        "OxiBonsai (CPU)"
    };
    tracing::info!("backend: {backend}, 1-bit Q1_0");

    // Resolve the GGUF chat model + tokenizer (local paths or download from HF).
    let model_path = resolve_model(&cfg)?;
    let tokenizer_path = resolve_tokenizer(&cfg)?;
    tracing::info!("loading chat model: {}", model_path.display());

    let init = llm::start(LlmConfig {
        model_path,
        tokenizer_path,
        max_seq_len: cfg.ctx_size as usize,
        temperature: cfg.temperature,
        top_p: cfg.top_p,
        top_k: cfg.top_k,
        repetition_penalty: cfg.repeat_penalty,
        seed: cfg.seed,
    })
    .context("failed to start inference worker")?;
    let env = Arc::new(chat::build_env(init.chat_template).context("invalid chat template")?);
    let bos_token: Arc<str> = Arc::from(init.bos_token.as_str());
    tracing::info!("chat model ready");

    // Load the embedding model (downloads on first run).
    tracing::info!("loading embedding model ({})", cfg.embed_id);
    let embedder = Arc::new(Embedder::new().context("failed to load embedding model")?);
    tracing::info!("embedding model ready (dim={})", embedder.dim);

    let state = AppState {
        llm: init.handle,
        embedder,
        env,
        bos_token,
        default_max_tokens: cfg.max_tokens,
        model_id: cfg.model_id.clone(),
        embed_id: cfg.embed_id.clone(),
    };

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    tracing::info!("listening on http://{addr}");
    tracing::info!("  POST /v1/chat/completions  (model: {})", cfg.model_id);
    tracing::info!("  POST /v1/embeddings        (model: {})", cfg.embed_id);

    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    Ok(())
}

fn resolve_model(cfg: &Config) -> Result<PathBuf> {
    if let Some(path) = &cfg.model {
        anyhow::ensure!(path.exists(), "model file not found: {}", path.display());
        return Ok(path.clone());
    }

    tracing::info!(
        "no --model given; fetching {} / {} from Hugging Face",
        cfg.model_repo,
        cfg.model_file
    );
    let api = hf_hub::api::sync::Api::new().context("failed to init Hugging Face client")?;
    let path = api
        .model(cfg.model_repo.clone())
        .get(&cfg.model_file)
        .with_context(|| {
            format!(
                "failed to download {} from {}",
                cfg.model_file, cfg.model_repo
            )
        })?;
    Ok(path)
}

fn resolve_tokenizer(cfg: &Config) -> Result<PathBuf> {
    if let Some(path) = &cfg.tokenizer {
        anyhow::ensure!(path.exists(), "tokenizer file not found: {}", path.display());
        return Ok(path.clone());
    }

    tracing::info!(
        "no --tokenizer given; fetching {} / {} from Hugging Face",
        cfg.tokenizer_repo,
        cfg.tokenizer_file
    );
    let api = hf_hub::api::sync::Api::new().context("failed to init Hugging Face client")?;
    let path = api
        .model(cfg.tokenizer_repo.clone())
        .get(&cfg.tokenizer_file)
        .with_context(|| {
            format!(
                "failed to download {} from {}",
                cfg.tokenizer_file, cfg.tokenizer_repo
            )
        })?;
    Ok(path)
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
