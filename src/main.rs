mod api;
mod config;
mod embeddings;
mod gemma;
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

    tracing::info!(
        "acceleration: {} (gpu_layers={})",
        Config::acceleration(),
        cfg.gpu_layers()
    );

    // Resolve the GGUF chat model (local path or download from HF).
    let model_path = resolve_model(&cfg)?;
    tracing::info!("loading chat model: {}", model_path.display());

    let (llm, chat_template) = llm::start(LlmConfig {
        model_path,
        n_ctx: cfg.ctx_size,
        n_threads: cfg.threads(),
        n_gpu_layers: cfg.gpu_layers(),
    })
    .context("failed to start inference worker")?;
    let env = Arc::new(gemma::build_env(chat_template).context("invalid chat template")?);
    tracing::info!("chat model ready");

    // Load the embedding model (downloads on first run).
    tracing::info!("loading embedding model ({})", cfg.embed_id);
    let embedder = Arc::new(Embedder::new().context("failed to load embedding model")?);
    tracing::info!("embedding model ready (dim={})", embedder.dim);

    let state = AppState {
        llm,
        embedder,
        env,
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

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
