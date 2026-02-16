# allm-go

[![CI](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml/badge.svg)](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A thin, lightweight LLM client for Go. One API for Anthropic, OpenAI, DeepSeek, GLM (Zhipu AI), and local models.

```go
client := allm.New(provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")))
resp, _ := client.Complete(ctx, "Hello!")
fmt.Println(resp.Content)
```

## Features

- **5 providers** — Anthropic, OpenAI, DeepSeek, GLM (Zhipu AI), Local (Ollama/vLLM)
- **Chat & streaming** — multi-turn conversations with real SSE streaming
- **Vision** — send images to multimodal models
- **Embeddings** — generate text embeddings (OpenAI, GLM, Local)
- **Tool use** — function calling across all providers
- **Thread-safe** — concurrent-safe client with snapshot pattern
- **Retry** — configurable exponential backoff for transient errors
- **Logging & hooks** — structured logging and event callbacks for observability
- **Thin** — minimal abstraction, no unnecessary dependencies

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
    "os"

    "github.com/kusandriadi/allm-go"
    "github.com/kusandriadi/allm-go/provider"
)

func main() {
    client := allm.New(
        provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")),
        allm.WithModel(provider.AnthropicClaudeSonnet4_5),
        allm.WithMaxTokens(4096),
    )

    resp, err := client.Complete(context.Background(), "What is Go?")
    if err != nil {
        panic(err)
    }

    fmt.Println(resp.Content)
    fmt.Printf("Tokens: %d in, %d out\n", resp.InputTokens, resp.OutputTokens)
}
```

## Providers

```go
// Anthropic
client := allm.New(provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")))

// OpenAI
client := allm.New(provider.OpenAI(os.Getenv("OPENAI_API_KEY")))

// DeepSeek
client := allm.New(provider.DeepSeek(os.Getenv("DEEPSEEK_API_KEY")))

// GLM (Zhipu AI)
client := allm.New(provider.GLM(os.Getenv("GLM_API_KEY")))

// Local (Ollama, vLLM, any OpenAI-compatible server)
client := allm.New(provider.Ollama("llama3"))
client := allm.New(provider.VLLM("mistral"))
client := allm.New(provider.Local("http://my-server:8080/v1"))
```

Pass `""` to read from the default environment variable, or an explicit key to skip env lookup.

Custom base URL for proxies or Azure:

```go
client := allm.New(
    provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY"), provider.WithAnthropicBaseURL("https://proxy.example.com")),
)
client := allm.New(
    provider.OpenAI(os.Getenv("OPENAI_API_KEY"), provider.WithOpenAIBaseURL("https://your-resource.openai.azure.com/")),
    allm.WithModel("your-deployment"),
)
```

### Provider-Specific Options

| Provider | Options |
|----------|---------|
| Anthropic | `WithAnthropicModel`, `WithAnthropicMaxTokens`, `WithAnthropicTemperature`, `WithAnthropicBaseURL` |
| OpenAI | `WithOpenAIModel`, `WithOpenAIMaxTokens`, `WithOpenAITemperature`, `WithOpenAIBaseURL` |
| DeepSeek | `WithDeepSeekModel`, `WithDeepSeekMaxTokens`, `WithDeepSeekTemperature` |
| GLM | `WithGLMModel`, `WithGLMMaxTokens`, `WithGLMTemperature` |
| Local | `WithLocalModel`, `WithLocalAPIKey`, `WithLocalMaxTokens`, `WithLocalTemperature` |

## Usage

### Chat

```go
resp, _ := client.Chat(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "What is 2+2?"},
    {Role: allm.RoleAssistant, Content: "4"},
    {Role: allm.RoleUser, Content: "Multiply by 3?"},
})
```

### Streaming

```go
for chunk := range client.Stream(ctx, messages) {
    if chunk.Error != nil {
        panic(chunk.Error)
    }
    fmt.Print(chunk.Content)
}

// Or stream directly to any io.Writer
client.StreamToWriter(ctx, messages, os.Stdout)
```

### Vision

```go
imageData, _ := os.ReadFile("photo.jpg")

resp, _ := client.Chat(ctx, []allm.Message{
    {
        Role:    allm.RoleUser,
        Content: "What's in this image?",
        Images:  []allm.Image{{MimeType: "image/jpeg", Data: imageData}},
    },
})
```

### Embeddings

```go
// OpenAI, GLM, and Local providers support embeddings
client := allm.New(
    provider.OpenAI(os.Getenv("OPENAI_API_KEY")),
    allm.WithEmbeddingModel(provider.OpenAITextEmbedding3Small),
)

resp, _ := client.Embed(ctx, "Hello world")
fmt.Println(resp.Embeddings[0]) // []float64 vector

// Batch embedding
resp, _ = client.Embed(ctx, "Hello", "World", "Foo")
// resp.Embeddings has 3 vectors
```

Supported by: OpenAI, GLM, Local (Ollama/vLLM). Not supported by: Anthropic, DeepSeek.

### Tool Use (Function Calling)

```go
// 1. Define tools
client := allm.New(
    provider.OpenAI(os.Getenv("OPENAI_API_KEY")),
    allm.WithTools(allm.Tool{
        Name:        "get_weather",
        Description: "Get current weather for a city",
        Parameters: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "city": map[string]any{"type": "string", "description": "City name"},
            },
            "required": []any{"city"},
        },
    }),
)

// 2. Send request — model may return tool calls
resp, _ := client.Complete(ctx, "What's the weather in Tokyo?")

if len(resp.ToolCalls) > 0 {
    tc := resp.ToolCalls[0]
    fmt.Printf("Call: %s(%s)\n", tc.Name, tc.Arguments)

    // 3. Execute tool and send result back
    resp, _ = client.Chat(ctx, []allm.Message{
        {Role: allm.RoleUser, Content: "What's the weather in Tokyo?"},
        {Role: allm.RoleAssistant, ToolCalls: resp.ToolCalls},
        {Role: allm.RoleTool, ToolResults: []allm.ToolResult{
            {ToolCallID: tc.ID, Content: `{"temp": 22, "condition": "sunny"}`},
        }},
    })
    fmt.Println(resp.Content) // "The weather in Tokyo is 22°C and sunny."
}
```

Supported by all providers: Anthropic, OpenAI, DeepSeek, GLM, Local.

### Switch Model at Runtime

```go
client.SetModel(provider.AnthropicClaudeHaiku4_5)  // fast & cheap
client.SetModel(provider.AnthropicClaudeOpus4_6)   // powerful

client.SetProvider(provider.OpenAI(os.Getenv("OPENAI_API_KEY")))  // switch provider entirely
client.SetSystemPrompt("You are a pirate.")
client.SetTools(allm.Tool{Name: "search", Description: "Search"})
```

### List Models

```go
models, _ := client.Models(ctx)
for _, m := range models {
    fmt.Printf("%s (%s)\n", m.Name, m.ID)
}
```

## Client Options

```go
client := allm.New(p,
    allm.WithModel(provider.AnthropicClaudeSonnet4_5),
    allm.WithMaxTokens(8192),
    allm.WithTemperature(0.7),
    allm.WithPresencePenalty(0.1),          // -2.0 to 2.0
    allm.WithFrequencyPenalty(0.1),         // -2.0 to 2.0
    allm.WithTimeout(30 * time.Second),     // default: 60s
    allm.WithMaxInputLen(200000),           // default: 100KB
    allm.WithSystemPrompt("Be helpful."),
    allm.WithEmbeddingModel(provider.OpenAITextEmbedding3Small),
    allm.WithTools(tools...),               // function calling tools
    allm.WithMaxRetries(3),                 // retry transient errors (default: 0)
    allm.WithRetryBaseDelay(1 * time.Second),  // initial backoff (default: 1s)
    allm.WithRetryMaxDelay(30 * time.Second),  // max backoff (default: 30s)
    allm.WithLogger(slog.Default()),        // structured logging
    allm.WithHook(metricsHook),             // event callback
)
```

## Retry

Retry with exponential backoff for transient errors (rate limits, timeouts, empty responses). Retries apply to `Chat`/`Complete` and `Embed`, not `Stream`.

```go
client := allm.New(
    provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")),
    allm.WithMaxRetries(3),  // up to 3 retries (4 total attempts)
)
```

Each attempt gets its own timeout. Backoff follows `min(base * 2^attempt, max)` with 0-25% jitter. The default base delay is 1s and max is 30s.

Non-retryable errors (`ErrNoProvider`, `ErrEmptyInput`, `ErrInputTooLong`, `ErrCanceled`) fail immediately.

## Logging & Hooks

### Structured Logging

Pass any `*slog.Logger` (or any type with `Info`, `Warn`, `Error` methods):

```go
client := allm.New(
    provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")),
    allm.WithLogger(slog.Default()),
)
// Logs: request success/failure, retries with delay and error
```

### Event Hooks

Hook into client events for custom metrics or monitoring:

```go
client := allm.New(
    provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")),
    allm.WithHook(func(e allm.HookEvent) {
        switch e.Type {
        case allm.HookRequest:
            fmt.Printf("→ %s/%s attempt %d\n", e.Provider, e.Model, e.Attempt)
        case allm.HookSuccess:
            fmt.Printf("✓ %d+%d tokens in %v\n", e.InputTokens, e.OutputTokens, e.Latency)
        case allm.HookRetry:
            fmt.Printf("⟳ retry #%d: %v\n", e.Attempt, e.Error)
        case allm.HookError:
            fmt.Printf("✗ failed: %v\n", e.Error)
        }
    }),
)
```

## Model Constants

```go
// Anthropic
provider.AnthropicClaudeOpus4_6    // claude-opus-4-6
provider.AnthropicClaudeSonnet4_5  // claude-sonnet-4-5-20250929
provider.AnthropicClaudeHaiku4_5   // claude-haiku-4-5-20251001

// OpenAI
provider.OpenAIGPT5_2             // gpt-5.2
provider.OpenAIGPT4o              // gpt-4o
provider.OpenAIO3                 // o3
provider.OpenAIO4Mini             // o4-mini

// DeepSeek
provider.DeepSeekChat             // deepseek-chat
provider.DeepSeekReasoner         // deepseek-reasoner

// GLM (Zhipu AI)
provider.GLM5                     // glm-5
provider.GLM4_7                   // glm-4-7
provider.GLM4_7Flash              // glm-4-7-flash
provider.GLM4_5Flash              // glm-4-5-flash

// Embedding Models
provider.OpenAITextEmbedding3Small  // text-embedding-3-small
provider.OpenAITextEmbedding3Large  // text-embedding-3-large
provider.GLMEmbedding3              // embedding-3
```

## Testing

The `allmtest` package provides a mock provider for testing without real API calls.

```go
import "github.com/kusandriadi/allm-go/allmtest"

func TestMyService(t *testing.T) {
    mock := allmtest.NewMockProvider("test",
        allmtest.WithResponse(&allm.Response{Content: "Hello!"}),
    )
    client := allm.New(mock)

    resp, _ := client.Complete(ctx, "Hi")
    // resp.Content == "Hello!"

    mock.LastRequest()   // inspect last request
    mock.CallCount()     // number of calls
    mock.Requests()      // all requests
    mock.Reset()         // clear recorded requests
}
```

Mock errors, streaming, and embeddings:

```go
mock := allmtest.NewMockProvider("test", allmtest.WithError(errors.New("api down")))

mock := allmtest.NewMockProvider("test", allmtest.WithStreamChunks([]allm.StreamChunk{
    {Content: "Hello"}, {Content: " world"}, {Done: true},
}))

mock := allmtest.NewMockProvider("test", allmtest.WithEmbedResponse(&allm.EmbedResponse{
    Embeddings: [][]float64{{0.1, 0.2, 0.3}},
}))
```

## Error Handling

All errors use sentinel values for easy classification with `errors.Is`:

```go
resp, err := client.Complete(ctx, prompt)
if err != nil {
    switch {
    case errors.Is(err, allm.ErrEmptyInput):    // empty prompt
    case errors.Is(err, allm.ErrInputTooLong):  // exceeds max length
    case errors.Is(err, allm.ErrRateLimited):   // provider rate limit (429)
    case errors.Is(err, allm.ErrEmptyResponse):  // provider returned no content
    case errors.Is(err, allm.ErrTimeout):        // request timed out
    case errors.Is(err, allm.ErrCanceled):       // context canceled
    case errors.Is(err, allm.ErrNoProvider):     // nil provider
    case errors.Is(err, allm.ErrProvider):       // generic provider error
    default:                                      // provider-specific error
    }
}
```

## API Reference

### Client Methods

| Method | Description |
|--------|-------------|
| `Complete(ctx, prompt)` | Simple text completion |
| `Chat(ctx, messages)` | Multi-turn conversation |
| `Stream(ctx, messages)` | Streaming via channel |
| `StreamToWriter(ctx, messages, w)` | Stream to `io.Writer` |
| `Embed(ctx, texts...)` | Generate embeddings |
| `Models(ctx)` | List available models |
| `Provider()` | Get current provider |
| `SetModel(m)` | Switch model |
| `SetProvider(p)` | Switch provider |
| `SetSystemPrompt(s)` | Update system prompt |
| `SetTools(tools...)` | Update available tools |

### Types

| Type | Description |
|------|-------------|
| `Message` | Chat message with Role, Content, Images, ToolCalls, ToolResults |
| `Response` | LLM response with Content, ToolCalls, Provider, Model, token counts, Latency, FinishReason |
| `StreamChunk` | Streaming chunk with Content, Done, Error |
| `Image` | Image attachment with MimeType and Data |
| `Tool` | Tool definition with Name, Description, Parameters |
| `ToolCall` | Model's function call with ID, Name, Arguments |
| `ToolResult` | Tool execution result with ToolCallID, Content, IsError |
| `EmbedRequest` | Embedding input with texts and model |
| `EmbedResponse` | Embedding vectors with Model, Provider, InputTokens, Latency |
| `Model` | Model info with ID, Name, Provider |

### Interfaces

| Interface | Description |
|-----------|-------------|
| `Provider` | Core interface: Name, Complete, Stream, Available |
| `ModelLister` | Optional: Models(ctx) for listing models |
| `Embedder` | Optional: Embed(ctx, req) for embeddings |

## Provider Feature Matrix

| Feature | Anthropic | OpenAI | DeepSeek | GLM | Local |
|---------|:---------:|:------:|:--------:|:---:|:-----:|
| Chat | Yes | Yes | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes | Yes | Yes |
| Vision | Yes | Yes | - | - | - |
| Embeddings | - | Yes | - | Yes | Yes |
| Tool Use | Yes | Yes | Yes | Yes | Yes |
| Model Listing | Yes | Yes | Yes | Yes | Yes |
| Custom Base URL | Yes | Yes | - | - | Yes |

## Concurrency

`Client` is safe for concurrent use. All methods are protected by a read-write mutex using a snapshot pattern — you can call `SetModel` from one goroutine while `Chat` runs in another without data races.

## License

MIT
