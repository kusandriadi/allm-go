# allm-go

[![CI](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml/badge.svg)](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.26+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Lightweight Go client for LLMs. One interface, many providers.

```go
client := allm.New(provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")))
resp, _ := client.Complete(ctx, "Hello!")
fmt.Println(resp.Content)
```

## Features

- **7 providers** — Anthropic, OpenAI, GLM, Kimi, MiniMax, Local (Ollama/vLLM), Claude CLI
- **Any OpenAI-compatible API** via `OpenAICompatible()` — add new providers in one line
- **Chat, streaming, vision, embeddings, tool use** — same API across all providers
- **Adaptive effort** — `WithEffort("high")` for thinking/reasoning across providers (Anthropic, Ollama, OpenAI)
- **Extended thinking** — fine-grained token budget control for reasoning models
- **PDF/Document input** — send PDFs and documents to models that support them (Anthropic)
- **Citations** — extract citations from model responses (Anthropic)
- **Audio TTS/STT** — text-to-speech and speech-to-text (OpenAI Whisper/TTS)
- **Structured output** — JSON mode and JSON Schema for guaranteed structured responses
- **Prompt caching** — reduce costs with Anthropic's cache control
- **Token counting** — pre-request token estimation (Anthropic)
- **Context management** — automatic truncation when context exceeds limits
- **Image generation** — DALL-E support via OpenAI
- **Batch API** — submit bulk requests for async processing
- **Log probabilities** — per-token log probabilities (OpenAI/compatible)
- **Reproducibility** — seed parameter for deterministic outputs (OpenAI/compatible)
- **Thread-safe** — use one client from multiple goroutines
- **Retry with backoff** — automatic retry on rate limits (429), server errors (5xx), and overloaded (529)
- **Testing utilities** — mock provider and `allmtest.Verify()` for integration tests
- **Security** — SSRF protection, input validation, API key leak detection, error sanitization

## Install

```bash
go get github.com/kusandriadi/allm-go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/kusandriadi/allm-go"
    "github.com/kusandriadi/allm-go/provider"
)

func main() {
    client := allm.New(
        provider.Anthropic(""),  // reads ANTHROPIC_API_KEY from env
        allm.WithMaxTokens(1024),
    )

    // Simple completion
    resp, _ := client.Complete(context.Background(), "What is Go?")
    fmt.Println(resp.Content)

    // Multi-turn chat
    resp, _ = client.Chat(context.Background(), []allm.Message{
        {Role: allm.RoleUser, Content: "My name is Alice."},
        {Role: allm.RoleAssistant, Content: "Nice to meet you, Alice!"},
        {Role: allm.RoleUser, Content: "What's my name?"},
    })
    fmt.Println(resp.Content)

    // Streaming
    for chunk := range client.Stream(context.Background(), []allm.Message{
        {Role: allm.RoleUser, Content: "Count to 5"},
    }) {
        fmt.Print(chunk.Content)
    }
}
```

## Providers

```go
provider.Anthropic(apiKey)       // Claude (Opus, Sonnet, Haiku)
provider.OpenAI(apiKey)          // GPT, o-series
provider.GLM(apiKey)             // Zhipu AI GLM (Anthropic-compatible)
provider.Kimi(apiKey)            // Moonshot AI Kimi (Anthropic-compatible)
provider.MiniMax(apiKey)         // MiniMax (Anthropic-compatible)
provider.Ollama("qwen3.5")      // Local — Ollama
provider.VLLM("mistral")        // Local — vLLM
provider.ClaudeCLI()             // Claude CLI (exec-based, uses local claude binary)
```

Pass `""` to read from environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GLM_API_KEY`, etc).

**Custom provider** — any OpenAI-compatible API:

```go
provider.OpenAICompatible("cerebras", apiKey,
    provider.WithBaseURL("https://api.cerebras.ai/v1"),
    provider.WithDefaultModel("llama-4-scout-17b"),
)
```

## Effort-Based Reasoning

Control thinking effort across providers with a single API:

```go
// Works with Anthropic, Ollama (qwen3.5, etc), and OpenAI reasoning models
client := allm.New(
    provider.Ollama("qwen3.5"),
    allm.WithEffort(allm.EffortHigh),
)

resp, _ := client.Complete(ctx, "Solve step by step: what is 23 * 47?")
fmt.Println("Thinking:", resp.Thinking)  // reasoning trace
fmt.Println("Answer:", resp.Content)     // final answer

// Streaming with thinking
for chunk := range client.Stream(ctx, messages) {
    if chunk.Thinking != "" {
        fmt.Print("[think] ", chunk.Thinking)
    }
    if chunk.Content != "" {
        fmt.Print(chunk.Content)
    }
}
```

Effort levels: `EffortLow`, `EffortMedium`, `EffortHigh`, `EffortMax` (Anthropic Opus only).

For Anthropic, effort maps to thinking budget tokens automatically. For Ollama/OpenAI, it maps to `reasoning_effort`. You can also use `WithThinking(budgetTokens)` for fine-grained control.

## Structured Output

```go
client := allm.New(p, allm.WithResponseFormat(&allm.ResponseFormat{
    Type: allm.ResponseFormatJSONSchema,
    Name: "person",
    Schema: map[string]any{
        "type": "object",
        "properties": map[string]any{
            "name": map[string]any{"type": "string"},
            "age":  map[string]any{"type": "integer"},
        },
        "required": []any{"name", "age"},
    },
}))
resp, _ := client.Complete(ctx, "Generate a person")
// resp.Content is guaranteed valid JSON: {"name": "Alice", "age": 30}
```

## Prompt Caching

Reduce costs by caching system prompts and long context (Anthropic):

```go
resp, _ := client.Chat(ctx, []allm.Message{
    {
        Role:         allm.RoleSystem,
        Content:      longSystemPrompt, // cached on repeat calls
        CacheControl: &allm.CacheControl{Type: allm.CacheEphemeral},
    },
    {Role: allm.RoleUser, Content: "Hello"},
})
fmt.Printf("Cache read: %d tokens, Cache write: %d tokens\n",
    resp.CacheReadTokens, resp.CacheWriteTokens)
```

## Vision & Embeddings

```go
// Vision (Anthropic, OpenAI, GLM, Kimi)
resp, _ := client.Chat(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "Describe this image",
     Images: []allm.Image{{MimeType: "image/jpeg", Data: imgBytes}}},
})

// Embeddings (OpenAI, Local)
resp, _ := client.Embed(ctx, "Hello world")
```

## PDF/Document Input

Send PDFs and documents to models that support them (Anthropic):

```go
pdfBytes, _ := os.ReadFile("document.pdf")
resp, _ := client.Chat(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "Summarize this document",
     Documents: []allm.Document{{MimeType: "application/pdf", Data: pdfBytes, Name: "document.pdf"}}},
})
```

## Tool Use

```go
client := allm.New(p, allm.WithTools(allm.Tool{
    Name:        "get_weather",
    Description: "Get weather for a city",
    Parameters:  map[string]any{
        "type": "object",
        "properties": map[string]any{
            "city": map[string]any{"type": "string"},
        },
        "required": []any{"city"},
    },
}))

resp, _ := client.Complete(ctx, "Weather in Tokyo?")
for _, tc := range resp.ToolCalls {
    fmt.Printf("Call: %s(%s)\n", tc.Name, tc.Arguments)
}
```

## Audio (TTS/STT)

```go
// Text-to-speech
resp, _ := client.Speak(ctx, &allm.SpeechRequest{
    Input: "Hello, this is a test",
    Model: "tts-1-hd",
    Voice: "alloy",
})
os.WriteFile("output.mp3", resp.Audio, 0644)

// Speech-to-text (Whisper)
audioBytes, _ := os.ReadFile("audio.mp3")
resp, _ := client.Transcribe(ctx, &allm.TranscribeRequest{
    Audio:    audioBytes,
    Model:    "whisper-1",
    Language: "en",
})
fmt.Println(resp.Text)
```

## Image Generation

```go
resp, _ := client.GenerateImage(ctx, "A sunset over mountains",
    allm.WithImageModel("dall-e-3"),
    allm.WithImageSize(allm.ImageSize1024),
    allm.WithImageQuality("hd"),
)
fmt.Println("Image URL:", resp.Images[0].URL)
```

## Context Window Management

```go
client := allm.New(p,
    allm.WithMaxContextTokens(100000),
    allm.WithTruncationStrategy(allm.TruncateTail),
)
// If messages exceed 100k tokens, oldest non-system messages are removed
resp, _ := client.Chat(ctx, longConversation)
```

## Options

```go
allm.WithModel("opus")                      // model override
allm.WithMaxTokens(8192)                    // max output tokens
allm.WithTemperature(0.7)                   // sampling temperature
allm.WithSystemPrompt("Be helpful.")        // system prompt
allm.WithTimeout(30 * time.Second)          // request timeout (default: 60s)
allm.WithMaxRetries(3)                      // retry with exponential backoff
allm.WithEffort(allm.EffortHigh)            // reasoning effort level
allm.WithThinking(10000)                    // explicit thinking budget (tokens)
allm.WithResponseFormat(...)                // structured output (JSON mode/schema)
allm.WithMaxContextTokens(100000)           // context window limit
allm.WithLogger(slog.Default())             // structured logging
allm.WithLogProbs(5)                        // log probabilities (OpenAI/compatible)
allm.WithSeed(42)                           // reproducible outputs
allm.WithHook(func(e allm.HookEvent) {...}) // lifecycle events
```

Runtime updates: `SetModel()`, `SetProvider()`, `SetSystemPrompt()`, `SetTools()`, `SetResponseFormat()`, `SetThinking()`, `SetEffort()`.

## Testing

```go
// Mock provider for unit tests
mock := allmtest.NewMockProvider("test",
    allmtest.WithResponse(&allm.Response{Content: "Hello!"}),
)
client := allm.New(mock)
resp, _ := client.Complete(ctx, "Hi")  // returns "Hello!"

// Integration test
func TestMyProvider(t *testing.T) {
    client := allm.New(provider.OpenAI(""), allm.WithMaxTokens(256))
    allmtest.Verify(t, client)
}
```

## Error Handling

```go
resp, err := client.Complete(ctx, "Hello")
if errors.Is(err, allm.ErrRateLimited) {
    // back off and retry
}
if errors.Is(err, allm.ErrNotSupported) {
    // provider doesn't support this feature
}
```

Sentinel errors: `ErrRateLimited`, `ErrServerError`, `ErrOverloaded`, `ErrTimeout`, `ErrInputTooLong`, `ErrEmptyInput`, `ErrNoProvider`, `ErrEmptyResponse`, `ErrCanceled`, `ErrProvider`, `ErrNotSupported`.

## Feature Matrix

| | Anthropic | OpenAI | GLM | Kimi | MiniMax | Local (Ollama) | Claude CLI |
|--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Chat | Y | Y | Y | Y | Y | Y | Y |
| Streaming | Y | Y | Y | Y | Y | Y | Y |
| Vision | Y | Y | Y | Y | | Y | |
| Embeddings | | Y | | | | Y | |
| Tool Use | Y | Y | Y | Y | Y | Y | |
| Thinking/Effort | Y | Y | | | | Y | Y |
| Structured Output | | Y | | | | Y | |
| Prompt Caching | Y | | | | | | |
| Token Counting | Y | | | | | | |
| Image Generation | | Y | | | | | |
| Models List | Y | Y | Y | Y | Y | Y | Y |

## License

MIT
