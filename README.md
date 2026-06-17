# tinymodelserver

A tiny, fast, low-memory OpenAI-compatible server (Rust) that hosts two local models:

- **Chat** — PrismML **Bonsai-8B**, a 1-bit (`Q1_0`) model run by the pure-Rust
  [OxiBonsai](https://github.com/cool-japan/oxibonsai) engine — ~1.1 GB on disk, <2 GB RAM,
  8B-class quality.
- **Embeddings** — `multilingual-e5-small` via [fastembed](https://crates.io/crates/fastembed) (multilingual, 384-dim).

Binds to **localhost only** on a configurable port. **Pure Rust — no C/C++, no cmake.**
Metal-accelerated on macOS, CPU elsewhere.

> **Tool calling & templating.** The prompt is rendered from the model's *own* chat template
> (read from the GGUF metadata) via minijinja, so OpenAI `tools` are injected in the model's
> native format. Tool-call *output* parsing recognizes Qwen/Hermes
> (`<tool_call>{…}</tool_call>`, Bonsai's format), LFM2 Pythonic
> (`<|tool_call_start|>[fn(arg=…)]`), and Gemma (`<|tool_call>call:…`) conventions.

## Prerequisites

Managed with [mise](https://mise.jdx.dev):

```sh
mise trust && mise install      # installs rust (that's it — pure Rust build)
```

## Build & run

```sh
mise run build                  # cargo build --release
mise run run                    # runs on http://127.0.0.1:8080
```

On first launch it downloads the Bonsai GGUF (~1.1 GB), its `tokenizer.json`, and the
embedding model into the Hugging Face cache. Subsequent launches are offline-capable.

### Acceleration

Metal is enabled automatically on **macOS** (≈38 tok/s for Bonsai-8B on an M2 Max vs
≈5 tok/s CPU). On other platforms it runs on CPU (SIMD). The active backend is logged at
startup:

```
INFO backend: OxiBonsai (Metal), 1-bit Q1_0
```

### Configuration

All flags have `TMS_*` env-var equivalents:

| Flag | Default | Purpose |
|------|---------|---------|
| `--port` | `8080` | Listen port |
| `--host` | `127.0.0.1` | Bind address (localhost only) |
| `--model` | _(download)_ | Path to a local `.gguf` to skip the download |
| `--model-repo` / `--model-file` | `prism-ml/Bonsai-8B-gguf` / `Bonsai-8B-Q1_0.gguf` | HF source for the GGUF |
| `--tokenizer` | _(download)_ | Path to a local `tokenizer.json` |
| `--tokenizer-repo` / `--tokenizer-file` | `prism-ml/Bonsai-8B-unpacked` / `tokenizer.json` | HF source for the tokenizer |
| `--ctx-size` | `65536` | Max sequence length (Bonsai-8B's full 64K); sizes the KV buffers, and generation is capped so prompt + output fits |
| `--max-tokens` | `2048` | Default generation cap when the request omits `max_tokens` |
| `--temperature` / `--top-p` / `--top-k` / `--repeat-penalty` / `--seed` | `0.7` / `0.9` / `40` / `1.1` / `42` | Sampling (set at startup) |

Example with local files and a custom port:

```sh
./target/release/tinymodelserver --port 9000 \
  --model ~/models/Bonsai-8B-Q1_0.gguf --tokenizer ~/models/tokenizer.json
```

## API

OpenAI-compatible. Point any OpenAI client at `http://127.0.0.1:8080/v1`.

```sh
# Chat
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-8b","messages":[{"role":"user","content":"Hi in French?"}]}'

# Embeddings
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"multilingual-e5-small","input":["hello","bonjour"]}'

# Models / health
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/health
```

Supported chat params: `messages`, `max_tokens`, `tools`. (`temperature`/`top_p`/`top_k`/`seed`
are accepted but ignored — sampling is fixed at startup; see [Configuration](#configuration).)

### Function / tool calling

Standard OpenAI function calling is supported. Pass `tools`; when the model decides to
call one, the response comes back with `finish_reason: "tool_calls"` and a structured
`message.tool_calls` array (arguments as a JSON string), and you feed `role: "tool"`
results back in the next request — the normal agentic loop.

Prompts are rendered with the model's **own** chat template (read from the GGUF metadata and
run through minijinja), so tool definitions use whatever format the model expects. The
model's tool-call output is parsed back into OpenAI `tool_calls` for Qwen/Hermes
(`<tool_call>{…}</tool_call>`, Bonsai's format), LFM2 Pythonic (`<|tool_call_start|>[fn(arg=…)]`),
and Gemma (`<|tool_call>call:…`) conventions. Reasoning (`<think>…</think>`) is stripped.

### OpenAI SDK compatibility

Works directly with the official OpenAI SDKs — just point `base_url` at this server:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-noop")
c.embeddings.create(model="multilingual-e5-small", input=["hello", "bonjour"])
c.chat.completions.create(model="bonsai-8b",
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

- The non-`Send` OxiBonsai engine (and its mmap'd GGUF) lives on a dedicated worker thread;
  requests are queued over a channel, so the async HTTP layer stays clean and inference is
  serialized (one model in memory, predictable footprint).
- Embeddings run on a blocking thread pool.
- Streaming (`stream: true`) is not implemented; responses are returned whole.
