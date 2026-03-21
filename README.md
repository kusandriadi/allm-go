# allm-go

[![CI](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml/badge.svg)](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Lightweight Go client for LLMs. One interface, many providers.

```go
client := allm.New(provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")))
resp, _ := client.Complete(ctx, "Hello!")
fmt.Println(resp.Content)
```

## Features

- **9 providers** — Anthropic, OpenAI, DeepSeek, Gemini, GLM, Kimi, Qwen, MiniMax, Local (Ollama/vLLM)
- **Any OpenAI-compatible API** via `OpenAICompatible()` — add new providers in one line
- **Chat, streaming, vision, embeddings, tool use** — same API across all providers
- **Structured output** — JSON mode and JSON Schema for guaranteed structured responses
- **Extended thinking** — reasoning/chain-of-thought with budget control (Anthropic, DeepSeek)
- **Prompt caching** — reduce costs with Anthropic's cache control
- **Token counting** — pre-request token estimation (Anthropic)
- **Context management** — automatic truncation when context exceeds limits
- **Image generation** — DALL-E support via OpenAI
- **Batch API** — submit bulk requests for async processing
- **Thread-safe** — use one client from multiple goroutines
- **Retry with backoff** — automatic retry on rate limits and transient errors
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
provider.DeepSeek(apiKey)        // DeepSeek Chat/Reasoner
provider.Gemini(apiKey)          // Google Gemini
provider.GLM(apiKey)             // Zhipu AI GLM
provider.Kimi(apiKey)            // Moonshot AI Kimi
provider.Qwen(apiKey)            // Alibaba Qwen (DashScope)
provider.MiniMax(apiKey)         // MiniMax
provider.Ollama("llama3")        // Local — Ollama
provider.VLLM("mistral")        // Local — vLLM
```

Pass `""` to read from environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, etc).

**Custom provider** — any OpenAI-compatible API:

```go
provider.OpenAICompatible("cerebras", apiKey,
    provider.WithBaseURL("https://api.cerebras.ai/v1"),
    provider.WithDefaultModel("llama-4-scout-17b"),
)
```

## Structured Output

Force the model to return valid JSON matching a schema:

```go
// JSON mode — model outputs valid JSON
client := allm.New(p, allm.WithResponseFormat(&allm.ResponseFormat{
    Type: allm.ResponseFormatJSON,
}))

// JSON Schema — model outputs JSON matching your schema
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

## Extended Thinking

Enable reasoning/chain-of-thought for complex tasks:

```go
client := allm.New(
    provider.Anthropic(""),
    allm.WithModel("claude-sonnet-4-20250514"),
    allm.WithThinking(10000), // 10k token budget for thinking
    allm.WithMaxTokens(16000),
)

resp, _ := client.Complete(ctx, "Solve this step by step: ...")
fmt.Println("Thinking:", resp.Thinking)       // reasoning process
fmt.Println("Answer:", resp.Content)           // final answer
fmt.Println("Thinking tokens:", resp.ThinkingTokens)
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

## Token Counting

Estimate tokens before sending a request (Anthropic):

```go
count, _ := client.CountTokens(ctx, messages)
fmt.Printf("Input tokens: %d\n", count.InputTokens)
```

## Context Window Management

Automatically truncate conversations that exceed token limits:

```go
client := allm.New(p,
    allm.WithMaxContextTokens(100000),           // max context size
    allm.WithTruncationStrategy(allm.TruncateTail), // keep latest messages
)

// If messages exceed 100k tokens, oldest non-system messages are removed
resp, _ := client.Chat(ctx, longConversation)
```

## Vision & Embeddings

```go
// Vision (Anthropic, OpenAI, Gemini, GLM, Kimi, Qwen)
resp, _ := client.Chat(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "Describe this image",
     Images: []allm.Image{{MimeType: "image/jpeg", Data: imgBytes}}},
})

// Embeddings (OpenAI, GLM, Qwen, Local)
resp, _ := client.Embed(ctx, "Hello world")
```

## Image Generation

Generate images from text prompts (OpenAI DALL-E):

```go
resp, _ := client.GenerateImage(ctx, "A sunset over mountains",
    allm.WithImageModel("dall-e-3"),
    allm.WithImageSize(allm.ImageSize1024),
    allm.WithImageQuality("hd"),
)
fmt.Println("Image URL:", resp.Images[0].URL)
fmt.Println("Revised prompt:", resp.Images[0].RevisedPrompt)
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

## Options

```go
allm.WithModel("claude-opus-4-6")            // model override
allm.WithMaxTokens(8192)                     // max output tokens
allm.WithTemperature(0.7)                    // sampling temperature
allm.WithSystemPrompt("Be helpful.")         // system prompt
allm.WithTimeout(30 * time.Second)           // request timeout (default: 60s)
allm.WithMaxRetries(3)                       // retry with exponential backoff
allm.WithLogger(slog.Default())              // structured logging
allm.WithResponseFormat(&allm.ResponseFormat{// structured output
    Type: allm.ResponseFormatJSON,
})
allm.WithThinking(10000)                     // extended thinking budget
allm.WithMaxContextTokens(100000)            // context window limit
allm.WithTruncationStrategy(allm.TruncateTail) // auto-truncate
allm.WithHook(func(e allm.HookEvent) {      // lifecycle events
    fmt.Printf("%s latency=%v\n", e.Type, e.Latency)
})
```

Runtime updates: `SetModel()`, `SetProvider()`, `SetSystemPrompt()`, `SetTools()`, `SetResponseFormat()`, `SetThinking()`.

## Testing

```go
// Mock provider for unit tests
mock := allmtest.NewMockProvider("test",
    allmtest.WithResponse(&allm.Response{Content: "Hello!"}),
)
client := allm.New(mock)
resp, _ := client.Complete(ctx, "Hi")  // returns "Hello!"

// Integration test — verify a real provider works
func TestMyProvider(t *testing.T) {
    client := allm.New(provider.OpenAI(""), allm.WithMaxTokens(256))
    allmtest.Verify(t, client)  // tests chat, streaming, vision, embeddings, tools
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

Sentinel errors: `ErrRateLimited`, `ErrTimeout`, `ErrInputTooLong`, `ErrEmptyInput`, `ErrNoProvider`, `ErrEmptyResponse`, `ErrCanceled`, `ErrProvider`, `ErrNotSupported`.

## Feature Matrix

| | Anthropic | OpenAI | DeepSeek | Gemini | GLM | Kimi | Qwen | MiniMax | Local |
|--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Chat | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Streaming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Vision | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ | | ✓ |
| Embeddings | | ✓ | | | ✓ | | ✓ | | ✓ |
| Tool Use | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Models List | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Structured Output | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Extended Thinking | ✓ | | | | | | | | |
| Prompt Caching | ✓ | | | | | | | | |
| Token Counting | ✓ | | | | | | | | |
| Image Generation | | ✓ | | | | | | | |
| Batch API | | ✓* | | | | | | | |

\* Batch API interface defined; full implementation pending.

## License

MIT
