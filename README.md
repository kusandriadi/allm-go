# allm-go

[![CI](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml/badge.svg)](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A **thin**, **lightweight**, and **secure** LLM interface for Go. One API for Anthropic, OpenAI, DeepSeek, and local models.

```go
client := allm.New(provider.Anthropic(""))
resp, _ := client.Complete(ctx, "Hello!")
fmt.Println(resp.Content)
```

## Features

| Feature | Description |
|---------|-------------|
| **Thin** | Minimal abstraction, direct provider access |
| **Lightweight** | Only 2 direct dependencies (Anthropic SDK + OpenAI SDK) |
| **Secure** | Context support, input validation, safe defaults |
| **Simple** | Clean API, easy to understand |
| **Streaming** | First-class streaming support |
| **Multi-modal** | Image support for vision models |
| **Model listing** | Query available models from any provider |
| **Test helpers** | Built-in mock provider for integration testing |
| **Cross-platform** | Works on Linux, macOS, and Windows |

## Installation

```bash
go get github.com/kusandriadi/allm-go
```

## Quick Start

### Simple Completion

```go
package main

import (
    "context"
    "fmt"

    "github.com/kusandriadi/allm-go"
    "github.com/kusandriadi/allm-go/provider"
)

func main() {
    // Create client (reads ANTHROPIC_API_KEY from env)
    client := allm.New(provider.Anthropic(""))

    resp, err := client.Complete(context.Background(), "What is Go?")
    if err != nil {
        panic(err)
    }

    fmt.Println(resp.Content)
    fmt.Printf("Tokens: %d in, %d out\n", resp.InputTokens, resp.OutputTokens)
}
```

### Multi-turn Chat

```go
resp, _ := client.Chat(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "What is 2+2?"},
    {Role: allm.RoleAssistant, Content: "4"},
    {Role: allm.RoleUser, Content: "Multiply by 3?"},
})
```

### Streaming

```go
for chunk := range client.Stream(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "Tell me a story"},
}) {
    if chunk.Error != nil {
        panic(chunk.Error)
    }
    fmt.Print(chunk.Content)
}
```

### Stream to Writer

```go
err := client.StreamToWriter(ctx, messages, os.Stdout)
```

### List Available Models

```go
models, err := client.Models(ctx)
for _, m := range models {
    fmt.Printf("%s (%s)\n", m.Name, m.ID)
}
```

## Model Constants

Use built-in constants instead of hardcoding model ID strings. For runtime discovery, use `client.Models(ctx)`.

```go
// Anthropic
provider.AnthropicClaudeOpus4_6    // "claude-opus-4-6"
provider.AnthropicClaudeOpus4_5    // "claude-opus-4-5"
provider.AnthropicClaudeSonnet4_5  // "claude-sonnet-4-5-20250929"
provider.AnthropicClaudeSonnet4    // "claude-sonnet-4-20250514"
provider.AnthropicClaudeHaiku4_5   // "claude-haiku-4-5-20251001"

// OpenAI
provider.OpenAIGPT5_2             // "gpt-5.2"
provider.OpenAIGPT5_1             // "gpt-5.1"
provider.OpenAIGPT4_1             // "gpt-4.1"
provider.OpenAIGPT4o              // "gpt-4o"
provider.OpenAIO3                 // "o3"
provider.OpenAIO4Mini             // "o4-mini"

// DeepSeek
provider.DeepSeekChat             // "deepseek-chat"
provider.DeepSeekReasoner         // "deepseek-reasoner"
```

## Providers

### Anthropic (Claude)

```go
// Basic (reads ANTHROPIC_API_KEY from env, default model)
client := allm.New(provider.Anthropic(""))

// Recommended: model & params at client level
client := allm.New(
    provider.Anthropic("sk-ant-..."),
    allm.WithModel(provider.AnthropicClaudeOpus4_6),
    allm.WithMaxTokens(8192),
    allm.WithTemperature(0.7),
)

// Custom base URL (for proxies)
client := allm.New(
    provider.Anthropic("", provider.WithAnthropicBaseURL("https://proxy.example.com")),
    allm.WithModel(provider.AnthropicClaudeSonnet4_5),
)
```

### OpenAI (GPT)

```go
// Basic (reads OPENAI_API_KEY from env)
client := allm.New(provider.OpenAI(""))

// With model
client := allm.New(
    provider.OpenAI(""),
    allm.WithModel(provider.OpenAIGPT5_2),
    allm.WithMaxTokens(4096),
)

// Reasoning model
client := allm.New(
    provider.OpenAI(""),
    allm.WithModel(provider.OpenAIO3),
)

// Azure OpenAI
client := allm.New(
    provider.OpenAI("your-key",
        provider.WithOpenAIBaseURL("https://your-resource.openai.azure.com/"),
    ),
    allm.WithModel("your-deployment"),
)
```

### DeepSeek

```go
// Basic (reads DEEPSEEK_API_KEY from env)
client := allm.New(provider.DeepSeek(""))

// Reasoner model
client := allm.New(
    provider.DeepSeek(""),
    allm.WithModel(provider.DeepSeekReasoner),
)
```

### Local (Ollama, vLLM, etc.)

```go
// Ollama (default: localhost:11434)
client := allm.New(provider.Ollama("llama3"))

// vLLM (default: localhost:8000)
client := allm.New(provider.VLLM("mistral"))

// Custom OpenAI-compatible server
client := allm.New(
    provider.Local("http://my-server:8080/v1",
        provider.WithLocalAPIKey("optional-key"),
    ),
    allm.WithModel("phi-3"),
)
```

### Switch Model at Runtime

```go
// Same API key, different models — no need to recreate provider
client := allm.New(
    provider.Anthropic(""),
    allm.WithModel(provider.AnthropicClaudeOpus4_6),
)
resp, _ := client.Complete(ctx, "Complex task...")

client.SetModel(provider.AnthropicClaudeHaiku4_5)
resp, _ = client.Complete(ctx, "Simple task...")
```

## Client Options

```go
client := allm.New(p,
    allm.WithModel(provider.AnthropicClaudeSonnet4_5), // Model
    allm.WithMaxTokens(8192),                          // Max output tokens
    allm.WithTemperature(0.7),                         // Sampling temperature
    allm.WithTimeout(30 * time.Second),                // Request timeout (default: 60s)
    allm.WithMaxInputLen(200000),                      // Max input bytes (default: 100KB)
    allm.WithSystemPrompt("Be helpful."),              // System prompt
)
```

## Vision (Multi-modal)

```go
imageData, _ := os.ReadFile("photo.jpg")

resp, _ := client.Chat(ctx, []allm.Message{
    {
        Role:    allm.RoleUser,
        Content: "What's in this image?",
        Images: []allm.Image{
            {MimeType: "image/jpeg", Data: imageData},
        },
    },
})
```

## API Key Configuration

API keys can be provided in two ways:

| Provider | Parameter | Environment Variable |
|----------|-----------|---------------------|
| Anthropic | `provider.Anthropic("sk-ant-...")` | `ANTHROPIC_API_KEY` |
| OpenAI | `provider.OpenAI("sk-...")` | `OPENAI_API_KEY` |
| DeepSeek | `provider.DeepSeek("sk-...")` | `DEEPSEEK_API_KEY` |
| Local | `provider.WithLocalAPIKey("...")` | `LOCAL_API_KEY` |

Pass an empty string `""` to read from the environment variable automatically.

## Response

```go
type Response struct {
    Content      string        // Generated text
    Provider     string        // "anthropic", "openai", "deepseek", "local"
    Model        string        // Model used
    InputTokens  int           // Input token count
    OutputTokens int           // Output token count
    Latency      time.Duration // Request latency
    FinishReason string        // Why generation stopped
}
```

## Error Handling

```go
resp, err := client.Complete(ctx, prompt)
if err != nil {
    switch {
    case errors.Is(err, allm.ErrNoProvider):
        // No provider configured
    case errors.Is(err, allm.ErrEmptyInput):
        // Empty prompt
    case errors.Is(err, allm.ErrInputTooLong):
        // Input exceeds max length
    case errors.Is(err, allm.ErrTimeout):
        // Request timed out
    case errors.Is(err, allm.ErrCanceled):
        // Context canceled
    default:
        // Provider-specific error
    }
}
```

## Testing with allm-go

The `allmtest` package provides a `MockProvider` for integration testing. Use it to test services that depend on allm-go without making real API calls.

### Basic Mock

```go
import (
    "github.com/kusandriadi/allm-go"
    "github.com/kusandriadi/allm-go/allmtest"
)

func TestMyService(t *testing.T) {
    mock := allmtest.NewMockProvider("test",
        allmtest.WithResponse(&allm.Response{
            Content:  "Hello!",
            Provider: "test",
        }),
    )
    client := allm.New(mock)

    // Use client in your service
    resp, err := client.Complete(ctx, "Hi")
    // assert resp.Content == "Hello!"
}
```

### Inspect Requests

```go
func TestRequestCapture(t *testing.T) {
    mock := allmtest.NewMockProvider("test")
    client := allm.New(mock)

    client.Complete(ctx, "Hello")

    req := mock.LastRequest()       // last request sent
    count := mock.CallCount()       // number of calls
    allReqs := mock.Requests()      // all requests
}
```

### Mock Errors

```go
mock := allmtest.NewMockProvider("test",
    allmtest.WithError(errors.New("api down")),
)
```

### Mock Streaming

```go
mock := allmtest.NewMockProvider("test",
    allmtest.WithStreamChunks([]allm.StreamChunk{
        {Content: "Hello"},
        {Content: " world"},
        {Done: true},
    }),
)
```

### Dynamic Response Changes

```go
mock := allmtest.NewMockProvider("test")
mock.SetResponse(&allm.Response{Content: "first"})
client.Complete(ctx, "a")

mock.SetResponse(&allm.Response{Content: "second"})
client.Complete(ctx, "b")

mock.Reset() // clear all recorded requests
```

## Available APIs

| Method | Description |
|--------|-------------|
| `client.Complete(ctx, prompt)` | Simple text completion |
| `client.Chat(ctx, messages)` | Multi-turn conversation |
| `client.Stream(ctx, messages)` | Streaming response (channel) |
| `client.StreamToWriter(ctx, messages, w)` | Stream directly to io.Writer |
| `client.Models(ctx)` | List available models |
| `client.Provider()` | Get current provider |
| `client.SetProvider(p)` | Switch provider at runtime |
| `client.SetModel(m)` | Switch model at runtime |
| `client.SetSystemPrompt(s)` | Update system prompt |

## Security

- **Context support**: All operations respect `context.Context` for cancellation and timeouts
- **Input validation**: Max input length enforced (configurable, default 100KB)
- **Safe defaults**: Reasonable timeouts (60s), token limits (4096)
- **No credentials in logs**: API keys never logged
- **TLS by default**: All provider SDKs use HTTPS

## License

MIT License
