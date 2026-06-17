//! Chat inference backend: OxiBonsai (pure-Rust 1-bit engine) running a Bonsai
//! GGUF. The non-`Send` engine/model/mmap live on a dedicated worker thread;
//! requests arrive over a channel and results return via oneshot.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::mpsc;
use tokio::sync::oneshot;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;

/// Parameters for a single generation request.
///
/// Note: sampling (temperature/top_p/top_k/seed) is configured once at engine
/// build — OxiBonsai's blocking `generate()` uses the engine's sampler — so the
/// OpenAI request's per-call sampling fields are accepted but not applied.
pub struct GenRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

/// Result of a generation, including token accounting for the OpenAI usage block.
pub struct GenResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: &'static str,
}

struct Job {
    req: GenRequest,
    resp: oneshot::Sender<Result<GenResult, String>>,
}

/// Cheap, cloneable handle to the inference worker thread.
#[derive(Clone)]
pub struct LlmHandle {
    tx: mpsc::Sender<Job>,
}

impl LlmHandle {
    pub async fn generate(&self, req: GenRequest) -> Result<GenResult, String> {
        let (resp, rx) = oneshot::channel();
        self.tx
            .send(Job { req, resp })
            .map_err(|_| "inference worker is gone".to_string())?;
        rx.await
            .map_err(|_| "inference worker dropped the request".to_string())?
    }
}

pub struct LlmConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: u64,
}

/// What `start` returns once the model is ready.
pub struct LlmInit {
    pub handle: LlmHandle,
    /// The model's embedded Jinja chat template.
    pub chat_template: String,
    /// The model's BOS token as a string (empty if it has none, e.g. Qwen/Bonsai).
    pub bos_token: String,
}

/// Spawn the inference worker. Blocks until the model is loaded so startup
/// errors surface immediately rather than on the first request.
pub fn start(cfg: LlmConfig) -> Result<LlmInit> {
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(String, String), String>>();

    std::thread::Builder::new()
        .name("llm-worker".into())
        .spawn(move || {
            // mmap -> gguf -> engine form a borrow chain, so they all live here
            // as locals for the lifetime of the worker.
            let mmap = match mmap_gguf_file(&cfg.model_path) {
                Ok(m) => m,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("mmap {} failed: {e:?}", cfg.model_path.display())));
                    return;
                }
            };
            let gguf = match GgufFile::parse(&mmap[..]) {
                Ok(g) => g,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("GGUF parse failed: {e:?}")));
                    return;
                }
            };

            let template = gguf
                .metadata
                .get_string("tokenizer.chat_template")
                .unwrap_or_default()
                .to_string();

            let tokenizer = match TokenizerBridge::from_file(&cfg.tokenizer_path.to_string_lossy()) {
                Ok(t) => t,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("tokenizer load failed: {e:?}")));
                    return;
                }
            };

            let params = SamplingParams {
                temperature: cfg.temperature,
                top_k: cfg.top_k,
                top_p: cfg.top_p,
                repetition_penalty: cfg.repetition_penalty,
                max_tokens: cfg.max_seq_len,
            };

            let mut engine =
                match InferenceEngine::from_gguf(&gguf, params, cfg.seed, cfg.max_seq_len) {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("engine init failed: {e:?}")));
                        return;
                    }
                };

            // BOS: Bonsai/Qwen3 templates don't prepend one.
            let _ = ready_tx.send(Ok((template, String::new())));

            let max_seq_len = cfg.max_seq_len;
            for job in rx {
                let result = generate(&mut engine, &tokenizer, &job.req, max_seq_len)
                    .map_err(|e| e.to_string());
                let _ = job.resp.send(result);
            }
        })
        .context("failed to spawn inference worker thread")?;

    let (chat_template, bos_token) = ready_rx
        .recv()
        .context("inference worker exited during startup")?
        .map_err(anyhow::Error::msg)?;

    Ok(LlmInit {
        handle: LlmHandle { tx },
        chat_template,
        bos_token,
    })
}

fn generate(
    engine: &mut InferenceEngine,
    tokenizer: &TokenizerBridge,
    req: &GenRequest,
    max_seq_len: usize,
) -> Result<GenResult> {
    let prompt_tokens = tokenizer
        .encode(&req.prompt)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e:?}"))?;
    let n_prompt = prompt_tokens.len();

    // The engine's KV buffers are sized to max_seq_len; the position must never
    // reach it or OxiBonsai indexes out of bounds. Reject an over-long prompt
    // and cap generation so prompt + output fits the window.
    anyhow::ensure!(
        n_prompt < max_seq_len,
        "prompt is {n_prompt} tokens but the context window is {max_seq_len}; \
         raise --ctx-size or shorten the input"
    );
    let budget = max_seq_len - n_prompt;
    let max = req.max_tokens.min(budget).max(1);

    // OxiBonsai is young and can panic on edge cases; contain it to this request
    // (requires unwinding — see the release profile) so the worker survives.
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.generate(&prompt_tokens, max)
    }))
    .map_err(|_| anyhow::anyhow!("inference engine panicked while generating"))?
    .map_err(|e| anyhow::anyhow!("generation failed: {e:?}"))?;

    let text = tokenizer
        .decode(&out)
        .map_err(|e| anyhow::anyhow!("decode failed: {e:?}"))?;

    // generate() stops either at EOS or after the token budget; infer which.
    let finish_reason = if out.len() >= max { "length" } else { "stop" };

    Ok(GenResult {
        text,
        prompt_tokens: n_prompt,
        completion_tokens: out.len(),
        finish_reason,
    })
}
