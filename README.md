# allm-go

[![CI](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml/badge.svg)](https://github.com/kusandriadi/allm-go/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Thin LLM client for Go. One interface for multiple providers.

```go
client := allm.New(provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")))
resp, _ := client.Complete(ctx, "Hello!")
```

## Install

```bash
go get github.com/kusandriadi/allm-go
```

## Providers

```go
provider.Anthropic(apiKey)       // Claude
provider.OpenAI(apiKey)          // GPT
provider.DeepSeek(apiKey)        // DeepSeek
provider.Gemini(apiKey)          // Gemini
provider.Groq(apiKey)            // Groq
provider.GLM(apiKey)             // Zhipu AI
provider.Perplexity(apiKey)      // Perplexity
provider.Ollama("llama3")        // Local (Ollama)
provider.VLLM("mistral")        // Local (vLLM)
```

Pass `""` to read from environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc).

Any OpenAI-compatible API works out of the box:

```go
provider.OpenAICompatible("cerebras", apiKey,
    provider.WithBaseURL("https://api.cerebras.ai/v1"),
    provider.WithDefaultModel("llama-4-scout-17b"),
)
```

All providers except Anthropic and OpenAI are shortcuts for `OpenAICompatible` with pre-configured defaults.

## Usage

```go
// Completion
resp, _ := client.Complete(ctx, "What is Go?")

// Chat
resp, _ := client.Chat(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "Hello"},
    {Role: allm.RoleAssistant, Content: "Hi!"},
    {Role: allm.RoleUser, Content: "How are you?"},
})

// Streaming
for chunk := range client.Stream(ctx, messages) {
    fmt.Print(chunk.Content)
}

// Vision (Anthropic, OpenAI)
resp, _ := client.Chat(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "Describe this",
     Images: []allm.Image{{MimeType: "image/jpeg", Data: imgBytes}}},
})

// Embeddings (OpenAI, GLM, Local)
resp, _ := client.Embed(ctx, "Hello world")

// Tool use
client := allm.New(p, allm.WithTools(tools...))
resp, _ := client.Complete(ctx, "Weather in Tokyo?")
// handle resp.ToolCalls
```

## Options

```go
allm.WithModel("claude-opus-4-6")       // model override
allm.WithMaxTokens(8192)                // max output tokens
allm.WithTemperature(0.7)               // sampling temperature
allm.WithSystemPrompt("Be helpful.")    // system prompt
allm.WithTimeout(30 * time.Second)      // request timeout (default: 60s)
allm.WithMaxRetries(3)                  // retry with exponential backoff
allm.WithLogger(slog.Default())         // structured logging
allm.WithHook(func(e allm.HookEvent) {  // event callback
    fmt.Printf("%s %v\n", e.Type, e.Latency)
})
```

All settings can be changed at runtime via `SetModel()`, `SetProvider()`, `SetSystemPrompt()`, `SetTools()`.

## Provider Names

Type-safe constants in the `allm` package:

```go
allm.Anthropic      allm.Gemini       allm.Perplexity
allm.OpenAI         allm.Groq         allm.Local
allm.DeepSeek       allm.GLM
```

## Model Constants

Common models available as constants. See [`provider/models.go`](provider/models.go) for the full list.

```go
provider.AnthropicClaudeOpus4_6     provider.OpenAIGPT5_2
provider.AnthropicClaudeSonnet4_5   provider.OpenAIO3
provider.AnthropicClaudeHaiku4_5    provider.DeepSeekChat
provider.GeminiFlash2_0             provider.GroqLlama3_3_70B
```

## Testing

```go
mock := allmtest.NewMockProvider("test",
    allmtest.WithResponse(&allm.Response{Content: "Hello!"}),
)
client := allm.New(mock)
resp, _ := client.Complete(ctx, "Hi")  // "Hello!"
```

## Error Handling

Sentinel errors with `errors.Is`: `ErrRateLimited`, `ErrTimeout`, `ErrInputTooLong`, `ErrEmptyInput`, `ErrNoProvider`, `ErrEmptyResponse`, `ErrCanceled`, `ErrProvider`.

## Feature Matrix

| | Anthropic | OpenAI | DeepSeek | Gemini | Groq | GLM | Perplexity | Local |
|--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Chat | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Streaming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Vision | ✓ | ✓ | | | | | | |
| Embeddings | | ✓ | | | | ✓ | | ✓ |
| Tool Use | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Thread-safe. MIT License.
