use clap::Parser;
use std::path::PathBuf;

/// Tiny OpenAI-compatible local server: Bonsai 1-bit chat (OxiBonsai) + fastembed embeddings.
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
    #[arg(long, env = "TMS_MODEL_REPO", default_value = "prism-ml/Bonsai-8B-gguf")]
    pub model_repo: String,

    /// GGUF filename inside the repo (used with --model-repo).
    #[arg(long, env = "TMS_MODEL_FILE", default_value = "Bonsai-8B-Q1_0.gguf")]
    pub model_file: String,

    /// Path to a local tokenizer.json. If omitted, it is downloaded from Hugging Face.
    #[arg(long, env = "TMS_TOKENIZER")]
    pub tokenizer: Option<PathBuf>,

    /// HF repo for the tokenizer.json (OxiBonsai needs an HF tokenizer file;
    /// the GGUF doesn't carry one in a form it can load).
    #[arg(
        long,
        env = "TMS_TOKENIZER_REPO",
        default_value = "prism-ml/Bonsai-8B-unpacked"
    )]
    pub tokenizer_repo: String,

    /// tokenizer.json filename inside the tokenizer repo.
    #[arg(long, env = "TMS_TOKENIZER_FILE", default_value = "tokenizer.json")]
    pub tokenizer_file: String,

    /// Public model id reported by the API (the OpenAI `model` field).
    #[arg(long, env = "TMS_MODEL_ID", default_value = "bonsai-8b")]
    pub model_id: String,

    /// Maximum sequence length (context window) for the engine.
    #[arg(long, env = "TMS_CTX_SIZE", default_value_t = 8192)]
    pub ctx_size: u32,

    /// Default max generated tokens when the request doesn't set max_tokens.
    /// Bonsai is a reasoning model, so leave room for thinking + answer.
    #[arg(long, env = "TMS_MAX_TOKENS", default_value_t = 2048)]
    pub max_tokens: usize,

    /// Sampling temperature (set once at engine startup).
    #[arg(long, env = "TMS_TEMPERATURE", default_value_t = 0.7)]
    pub temperature: f32,

    /// Top-p nucleus sampling.
    #[arg(long, env = "TMS_TOP_P", default_value_t = 0.9)]
    pub top_p: f32,

    /// Top-k sampling.
    #[arg(long, env = "TMS_TOP_K", default_value_t = 40)]
    pub top_k: usize,

    /// Repetition penalty.
    #[arg(long, env = "TMS_REPEAT_PENALTY", default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// RNG seed.
    #[arg(long, env = "TMS_SEED", default_value_t = 42)]
    pub seed: u64,

    /// Embedding model id reported by the API.
    #[arg(long, env = "TMS_EMBED_ID", default_value = "multilingual-e5-small")]
    pub embed_id: String,
}
