# allm-go

[![CI](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml/badge.svg)](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A thin, lightweight LLM client for Go. One API for Anthropic, OpenAI, DeepSeek, and local models.

```go
client := allm.New(provider.Anthropic(""))
resp, _ := client.Complete(ctx, "Hello!")
fmt.Println(resp.Content)
```

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
// Anthropic — reads ANTHROPIC_API_KEY from env
client := allm.New(provider.Anthropic(""))

// OpenAI — reads OPENAI_API_KEY from env
client := allm.New(provider.OpenAI(""))

// DeepSeek — reads DEEPSEEK_API_KEY from env
client := allm.New(provider.DeepSeek(""))

// Local (Ollama, vLLM, any OpenAI-compatible server)
client := allm.New(provider.Ollama("llama3"))
client := allm.New(provider.VLLM("mistral"))
client := allm.New(provider.Local("http://my-server:8080/v1"))
```

Pass an explicit key instead of `""` to skip env lookup: `provider.Anthropic("sk-ant-...")`.

Custom base URL for proxies or Azure:

```go
client := allm.New(
    provider.Anthropic("", provider.WithAnthropicBaseURL("https://proxy.example.com")),
)
client := allm.New(
    provider.OpenAI("key", provider.WithOpenAIBaseURL("https://your-resource.openai.azure.com/")),
    allm.WithModel("your-deployment"),
)
```

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

### Switch Model at Runtime

```go
client.SetModel(provider.AnthropicClaudeHaiku4_5)  // fast & cheap
client.SetModel(provider.AnthropicClaudeOpus4_6)   // powerful

client.SetProvider(provider.OpenAI(""))             // switch provider entirely
client.SetSystemPrompt("You are a pirate.")
```

### List Models

```go
models, _ := client.Models(ctx)
for _, m := range models {
    fmt.Printf("%s (%s)\n", m.Name, m.ID)
}
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
```

## Client Options

```go
client := allm.New(p,
    allm.WithModel(provider.AnthropicClaudeSonnet4_5),
    allm.WithMaxTokens(8192),
    allm.WithTemperature(0.7),
    allm.WithTimeout(30 * time.Second),    // default: 60s
    allm.WithMaxInputLen(200000),           // default: 100KB
    allm.WithSystemPrompt("Be helpful."),
)
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

Mock errors and streaming:

```go
mock := allmtest.NewMockProvider("test", allmtest.WithError(errors.New("api down")))

mock := allmtest.NewMockProvider("test", allmtest.WithStreamChunks([]allm.StreamChunk{
    {Content: "Hello"}, {Content: " world"}, {Done: true},
}))
```

## Error Handling

```go
resp, err := client.Complete(ctx, prompt)
if err != nil {
    switch {
    case errors.Is(err, allm.ErrEmptyInput):    // empty prompt
    case errors.Is(err, allm.ErrInputTooLong):  // exceeds max length
    case errors.Is(err, allm.ErrTimeout):       // request timed out
    case errors.Is(err, allm.ErrCanceled):      // context canceled
    default:                                     // provider-specific error
    }
}
```

## API Reference

| Method | Description |
|--------|-------------|
| `Complete(ctx, prompt)` | Simple text completion |
| `Chat(ctx, messages)` | Multi-turn conversation |
| `Stream(ctx, messages)` | Streaming via channel |
| `StreamToWriter(ctx, messages, w)` | Stream to `io.Writer` |
| `Models(ctx)` | List available models |
| `SetModel(m)` | Switch model |
| `SetProvider(p)` | Switch provider |
| `SetSystemPrompt(s)` | Update system prompt |

## Concurrency

`Client` is safe for concurrent use. All methods are protected by a read-write mutex. You can call `SetModel` from one goroutine while `Chat` runs in another.

## License

MIT
