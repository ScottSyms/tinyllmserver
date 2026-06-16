use clap::Parser;
use std::path::PathBuf;

/// Tiny OpenAI-compatible local server: Gemma 4 (GGUF) chat + fastembed embeddings.
#[derive(Parser, Debug, Clone)]
#[command(version, about)]
pub struct Config {
    /// Port to listen on.
    #[arg(short, long, env = "TMS_PORT", default_value_t = 8080)]
    pub port: u16,

    /// Host/interface to bind. Defaults to localhost only.
    #[arg(long, env = "TMS_HOST", default_value = "127.0.0.1")]
    pub host: String,

    /// Path to a local GGUF chat model. If omitted, it is downloaded from Hugging Face.
    #[arg(long, env = "TMS_MODEL")]
    pub model: Option<PathBuf>,

    /// Hugging Face repo to pull the GGUF from when --model is not set.
    #[arg(long, env = "TMS_MODEL_REPO", default_value = "unsloth/gemma-4-E2B-it-GGUF")]
    pub model_repo: String,

    /// GGUF filename inside the repo (used with --model-repo).
    #[arg(
        long,
        env = "TMS_MODEL_FILE",
        default_value = "gemma-4-E2B-it-Q4_K_M.gguf"
    )]
    pub model_file: String,

    /// Public model id reported by the API (the OpenAI `model` field).
    #[arg(long, env = "TMS_MODEL_ID", default_value = "gemma-4-e2b-it")]
    pub model_id: String,

    /// Context window size (tokens).
    #[arg(long, env = "TMS_CTX_SIZE", default_value_t = 4096)]
    pub ctx_size: u32,

    /// CPU threads for inference. Defaults to available parallelism.
    #[arg(long, env = "TMS_THREADS")]
    pub threads: Option<i32>,

    /// Layers to offload to GPU. Defaults: all on macOS (Metal), 0 elsewhere.
    #[arg(long, env = "TMS_GPU_LAYERS")]
    pub gpu_layers: Option<u32>,

    /// Embedding model id reported by the API.
    #[arg(long, env = "TMS_EMBED_ID", default_value = "multilingual-e5-small")]
    pub embed_id: String,
}

impl Config {
    pub fn threads(&self) -> i32 {
        self.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4)
        })
    }

    pub fn gpu_layers(&self) -> u32 {
        self.gpu_layers.unwrap_or(if cfg!(target_os = "macos") { 999 } else { 0 })
    }
}
