use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::sync::Mutex;

/// Wraps fastembed's `TextEmbedding`. `embed` needs `&mut self`, so calls are
/// serialized behind a Mutex; run them on a blocking thread from async code.
pub struct Embedder {
    inner: Mutex<TextEmbedding>,
    pub dim: usize,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        // Multilingual, 384-dim, small + fast. Downloads on first use.
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_show_download_progress(true),
        )
        .context("failed to initialize embedding model")?;
        Ok(Self {
            inner: Mutex::new(model),
            dim: 384,
        })
    }

    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut model = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("embedding model lock poisoned"))?;
        model.embed(texts, None).context("embedding failed")
    }
}
