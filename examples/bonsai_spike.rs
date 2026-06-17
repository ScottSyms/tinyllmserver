//! De-risking spike: load Bonsai-8B (Q1_0) via OxiBonsai, read the embedded
//! chat template, generate from a Qwen3-templated prompt, and report tok/s.
//!
//! Usage: cargo run --release --example bonsai_spike -- <model.gguf> <tokenizer.json>

use std::path::Path;
use std::time::Instant;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf_path = args.next().expect("usage: bonsai_spike <gguf> <tokenizer.json>");
    let tok_path = args.next().expect("usage: bonsai_spike <gguf> <tokenizer.json>");

    println!("mmap + parse GGUF: {gguf_path}");
    let mmap = mmap_gguf_file(Path::new(&gguf_path))?;
    let gguf = GgufFile::parse(&mmap[..]).map_err(|e| anyhow::anyhow!("gguf parse: {e:?}"))?;

    let arch = gguf.metadata.get_string("general.architecture").unwrap_or("?");
    let tmpl = gguf
        .metadata
        .get_string("tokenizer.chat_template")
        .unwrap_or("<none>");
    println!("architecture = {arch}");
    println!("chat_template length = {} chars", tmpl.len());

    println!("loading tokenizer: {tok_path}");
    let tok = TokenizerBridge::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tok: {e:?}"))?;
    println!("vocab_size = {}", tok.vocab_size());

    let params = SamplingParams {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.1,
        max_tokens: 256,
    };
    println!("building engine…");
    let mut engine = InferenceEngine::from_gguf(&gguf, params, 42, 4096)
        .map_err(|e| anyhow::anyhow!("engine: {e:?}"))?;

    // Minimal Qwen3/ChatML prompt.
    let prompt = "<|im_start|>user\nReply with exactly: hello world<|im_end|>\n<|im_start|>assistant\n";
    let ids = tok.encode(prompt).map_err(|e| anyhow::anyhow!("encode: {e:?}"))?;
    println!("prompt tokens = {}", ids.len());

    let t0 = Instant::now();
    let out = engine
        .generate(&ids, 256)
        .map_err(|e| anyhow::anyhow!("generate: {e:?}"))?;
    let dt = t0.elapsed().as_secs_f64();
    let text = tok.decode(&out).map_err(|e| anyhow::anyhow!("decode: {e:?}"))?;

    println!(
        "generated {} tokens in {:.2}s ({:.1} tok/s)",
        out.len(),
        dt,
        out.len() as f64 / dt
    );
    println!("---- OUTPUT ----\n{text}\n----------------");
    Ok(())
}
