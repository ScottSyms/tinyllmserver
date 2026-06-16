use anyhow::{Context, Result};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc;
use tokio::sync::oneshot;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

/// Parameters for a single generation request.
pub struct GenRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u32,
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
    pub n_ctx: u32,
    pub n_threads: i32,
    pub n_gpu_layers: u32,
}

/// Spawn the inference worker. Blocks until the model is loaded so startup
/// errors surface immediately rather than on the first request.
pub fn start(cfg: LlmConfig) -> Result<LlmHandle> {
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

    std::thread::Builder::new()
        .name("llm-worker".into())
        .spawn(move || {
            let backend = match LlamaBackend::init() {
                Ok(b) => b,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("backend init failed: {e}")));
                    return;
                }
            };

            let model_params = LlamaModelParams::default().with_n_gpu_layers(cfg.n_gpu_layers);
            let model = match LlamaModel::load_from_file(&backend, &cfg.model_path, &model_params) {
                Ok(m) => m,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("model load failed: {e}")));
                    return;
                }
            };

            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(cfg.n_ctx))
                .with_n_threads(cfg.n_threads)
                .with_n_threads_batch(cfg.n_threads);

            let mut ctx = match model.new_context(&backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("context creation failed: {e}")));
                    return;
                }
            };

            let _ = ready_tx.send(Ok(()));

            // Serve requests one at a time on this thread (state is not Send).
            for job in rx {
                let result = generate(&model, &mut ctx, cfg.n_ctx, &job.req)
                    .map_err(|e| e.to_string());
                let _ = job.resp.send(result);
            }
        })
        .context("failed to spawn inference worker thread")?;

    ready_rx
        .recv()
        .context("inference worker exited during startup")?
        .map_err(anyhow::Error::msg)?;

    Ok(LlmHandle { tx })
}

fn build_sampler(req: &GenRequest) -> LlamaSampler {
    if req.temperature <= 0.0 {
        return LlamaSampler::greedy();
    }
    LlamaSampler::chain_simple([
        LlamaSampler::top_k(req.top_k.max(1)),
        LlamaSampler::top_p(req.top_p.clamp(0.0, 1.0), 1),
        LlamaSampler::temp(req.temperature),
        LlamaSampler::dist(req.seed),
    ])
}

fn generate(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    n_ctx: u32,
    req: &GenRequest,
) -> Result<GenResult> {
    ctx.clear_kv_cache();

    let tokens = model
        .str_to_token(&req.prompt, AddBos::Always)
        .context("tokenization failed")?;
    let n_prompt = tokens.len();

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last = tokens.len() as i32 - 1;
    for (i, token) in tokens.into_iter().enumerate() {
        batch.add(token, i as i32, &[0], i as i32 == last)?;
    }
    ctx.decode(&mut batch).context("prompt decode failed")?;

    let mut sampler = build_sampler(req);
    let mut out_bytes: Vec<u8> = Vec::new();
    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0usize;
    let mut finish_reason = "length";

    while n_decode < req.max_tokens {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            finish_reason = "stop";
            break;
        }

        let bytes = model.token_to_piece_bytes(token, 32, false, None)?;
        out_bytes.extend_from_slice(&bytes);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        n_decode += 1;

        if n_cur as u32 >= n_ctx {
            finish_reason = "length";
            break;
        }
        ctx.decode(&mut batch).context("decode failed")?;
    }

    // Gemma's `<end_of_turn>` marker is not flagged as an EOG token, so it can
    // render as literal text just before generation stops. Cut it off.
    let mut text = String::from_utf8_lossy(&out_bytes).to_string();
    if let Some(idx) = text.find("<end_of_turn>") {
        text.truncate(idx);
        finish_reason = "stop";
    }
    let text = text.trim().to_string();

    Ok(GenResult {
        text,
        prompt_tokens: n_prompt,
        completion_tokens: n_decode,
        finish_reason,
    })
}
