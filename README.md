# tinymodelserver

A tiny, fast, low-memory OpenAI-compatible server (Rust) that hosts two local models:

- **Chat** — Gemma 4 E4B (instruction-tuned), 4-bit **GGUF** via [llama.cpp](https://github.com/ggml-org/llama.cpp) bindings.
- **Embeddings** — `multilingual-e5-small` via [fastembed](https://crates.io/crates/fastembed) (multilingual, 384-dim).

Binds to **localhost only** on a configurable port. Runs on **macOS** (Metal GPU) and **Windows** (CPU).

> **Note on the model:** an MLX `…-4bit` build of Gemma 4 is Apple-only and won't run on
> Windows. To get the *same model on both platforms*, this server uses the **GGUF** build
> (`gemma-4-E4B-it-Q4_K_M`, ~4-bit). It is Metal-accelerated on Mac and runs on CPU on Windows.
> E4B is the larger "effective 4B" variant — better at agentic tool use than E2B. Swap to
> the lighter E2B with `--model-repo unsloth/gemma-4-E2B-it-GGUF --model-file gemma-4-E2B-it-Q4_K_M.gguf`.

## Prerequisites

Managed with [mise](https://mise.jdx.dev):

```sh
mise trust && mise install      # installs rust + cmake
```

`cmake` and a C/C++ compiler are required to build the llama.cpp backend
(Xcode CLT on macOS; MSVC Build Tools on Windows).

## Build & run

```sh
mise run build                  # cargo build --release
mise run run                    # runs on http://127.0.0.1:8080
```

On first launch it downloads the chat GGUF (~5 GB) and the embedding model into the
Hugging Face cache. Subsequent launches are offline-capable.

### GPU acceleration

The default build is **Metal-accelerated on macOS** and **CPU elsewhere**. On
Windows/Linux with a GPU, enable the matching backend at build time (the relevant
toolkit must be installed):

```sh
# NVIDIA (CUDA toolkit required)
cargo build --release --features cuda

# Cross-vendor (Vulkan SDK required) — NVIDIA / AMD / Intel
cargo build --release --features vulkan

# AMD (ROCm/HIP required)
cargo build --release --features rocm
```

When an accelerated backend is compiled in, all model layers are offloaded to the GPU
by default (`--gpu-layers 999`) and flash attention is enabled automatically where the
backend supports it. The active backend is printed at startup:

```
INFO acceleration: CUDA (gpu_layers=999)
```

Tune offload with `--gpu-layers N` (lower it if the model doesn't fit in VRAM).

### Configuration

All flags have `TMS_*` env-var equivalents:

| Flag | Default | Purpose |
|------|---------|---------|
| `--port` | `8080` | Listen port |
| `--host` | `127.0.0.1` | Bind address (localhost only) |
| `--model` | _(download)_ | Path to a local `.gguf` to skip the download |
| `--model-repo` / `--model-file` | `unsloth/gemma-4-E4B-it-GGUF` / `gemma-4-E4B-it-Q4_K_M.gguf` | HF source |
| `--ctx-size` | `8192` | Context window (Gemma 4 supports up to 256K) |
| `--threads` | _auto_ | CPU threads |
| `--gpu-layers` | all (GPU build) / 0 (CPU) | Layers to offload to GPU |

Example with a local model and custom port:

```sh
./target/release/tinymodelserver --port 9000 --model ~/models/gemma-4-E4B-it-Q4_K_M.gguf
```

## API

OpenAI-compatible. Point any OpenAI client at `http://127.0.0.1:8080/v1`.

```sh
# Chat
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e4b-it","messages":[{"role":"user","content":"Hi in French?"}]}'

# Embeddings
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"multilingual-e5-small","input":["hello","bonjour"]}'

# Models / health
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/health
```

Supported chat params: `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `tools`.

### Function / tool calling

Standard OpenAI function calling is supported. Pass `tools`; when the model decides to
call one, the response comes back with `finish_reason: "tool_calls"` and a structured
`message.tool_calls` array (arguments as a JSON string), and you feed `role: "tool"`
results back in the next request — the normal agentic loop.

Prompts are rendered with the model's **own** chat template (extracted from the GGUF and
run through minijinja), so tool definitions use Gemma 4's native format and the model's
`<|tool_call>…<tool_call|>` output is parsed back into OpenAI `tool_calls`.

### OpenAI SDK compatibility

Works directly with the official OpenAI SDKs — just point `base_url` at this server:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-noop")
c.embeddings.create(model="multilingual-e5-small", input=["hello", "bonjour"])
c.chat.completions.create(model="gemma-4-e4b-it",
    messages=[{"role": "user", "content": "Hi"}])
```

The embeddings endpoint honors `encoding_format` (`float` and `base64`). This matters:
the OpenAI SDKs default to requesting `base64` and decode it client-side, so a server
that ignored it would break the stock SDK. Both formats return identical vectors.

> **Note on retrieval quality:** `multilingual-e5-small` was trained with `query:` /
> `passage:` input prefixes. The OpenAI embeddings API has no way to signal which is which,
> so all text is embedded uniformly. Vectors are valid and cosine-comparable, but for
> best asymmetric retrieval you'd prefix inputs yourself before sending them.

## Design notes

- The non-`Send` llama.cpp context lives on a dedicated worker thread; requests are
  queued over a channel, so the async HTTP layer stays clean and inference is serialized
  (one model in memory, predictable footprint).
- Embeddings run on a blocking thread pool.
- Streaming (`stream: true`) is not implemented; responses are returned whole.
