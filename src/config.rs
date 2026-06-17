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
    #[arg(long, env = "TMS_MODEL_REPO", default_value = "unsloth/gemma-4-E4B-it-GGUF")]
    pub model_repo: String,

    /// GGUF filename inside the repo (used with --model-repo).
    #[arg(
        long,
        env = "TMS_MODEL_FILE",
        default_value = "gemma-4-E4B-it-Q4_K_M.gguf"
    )]
    pub model_file: String,

    /// Public model id reported by the API (the OpenAI `model` field).
    #[arg(long, env = "TMS_MODEL_ID", default_value = "gemma-4-e4b-it")]
    pub model_id: String,

    /// Context window size (tokens).
    #[arg(long, env = "TMS_CTX_SIZE", default_value_t = 8192)]
    pub ctx_size: u32,

    /// CPU threads for inference. Defaults to available parallelism.
    #[arg(long, env = "TMS_THREADS")]
    pub threads: Option<i32>,

    /// Layers to offload to GPU. Defaults: all when built with a GPU backend
    /// (Metal on macOS, or --features cuda/vulkan/rocm), 0 on CPU-only builds.
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
        // Offload everything to the GPU when an accelerated backend is compiled
        // in; otherwise keep all layers on the CPU.
        self.gpu_layers
            .unwrap_or(if Self::gpu_available() { 999 } else { 0 })
    }

    /// Whether a GPU backend was compiled into this build.
    pub const fn gpu_available() -> bool {
        cfg!(target_os = "macos")
            || cfg!(feature = "cuda")
            || cfg!(feature = "rocm")
            || cfg!(feature = "vulkan")
    }

    /// Human-readable name of the active acceleration backend.
    pub const fn acceleration() -> &'static str {
        if cfg!(feature = "cuda") {
            "CUDA"
        } else if cfg!(feature = "rocm") {
            "ROCm/HIP"
        } else if cfg!(feature = "vulkan") {
            "Vulkan"
        } else if cfg!(target_os = "macos") {
            "Metal"
        } else {
            "CPU"
        }
    }
}
