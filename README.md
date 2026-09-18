# Adaptive Free LLM API

A self-hosted API that provides free access to multiple LLM models through g4f. It automatically ranks models by success rate and latency, adapting to find the best working providers over time.

## Features

- **Auto-adaptive model ranking** - tracks success/failure rates and latency per model
- **OpenAI-compatible API** - works with any tool that supports `/v1/chat/completions`
- **25+ models** - DeepSeek, Qwen, LLaMA, GPT-4o, Claude, Gemini, and more
- **Retry provider** - automatically tries multiple providers per model
- **Swagger docs** at `/docs`

## Quick Start

```bash
docker compose up -d
```

API runs on `http://0.0.0.0:8000`.

## Usage

```bash
# List models (ranked by performance)
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "auto"
  }'
```

## Updating Code

The `~/freeAIApi` folder is volume-mounted into the container. Edit files and restart:

```bash
docker compose restart
```

No rebuild required.
