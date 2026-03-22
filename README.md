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

- **10 providers** — Anthropic, OpenAI, DeepSeek, Gemini, GLM, Kimi, Qwen, MiniMax, Local (Ollama/vLLM), Claude CLI
- **Any OpenAI-compatible API** via `OpenAICompatible()` — add new providers in one line
- **Chat, streaming, vision, embeddings, tool use** — same API across all providers
- **PDF/Document input** — send PDFs and documents to models that support them (Anthropic)
- **Citations** — extract citations from model responses (Anthropic)
- **Audio TTS/STT** — text-to-speech and speech-to-text (OpenAI Whisper/TTS)
- **Streaming token usage** — get token counts in streaming responses
- **Model metadata** — context window, max output, capabilities in model listings
- **Web search grounding** — built-in web search tool (provider-dependent)
- **Computer use** — tool definitions for Anthropic's computer use capability
- **Structured output** — JSON mode and JSON Schema for guaranteed structured responses
- **Extended thinking** — reasoning/chain-of-thought with budget control (Anthropic, DeepSeek)
- **Prompt caching** — reduce costs with Anthropic's cache control
- **Token counting** — pre-request token estimation (Anthropic)
- **Context management** — automatic truncation when context exceeds limits
- **Image generation** — DALL-E support via OpenAI
- **Batch API** — submit bulk requests for async processing
- **Log probabilities** — per-token log probabilities for confidence scoring (OpenAI/compatible)
- **Reproducibility** — seed parameter for deterministic outputs (OpenAI/compatible)
- **Parallel tool calls** — control parallel tool calling behavior (OpenAI/compatible)
- **Predicted output** — efficient editing with predicted content (OpenAI)
- **Request tracking** — capture provider request IDs for debugging
- **Thread-safe** — use one client from multiple goroutines
- **Health check** — `Ping()` verifies provider connectivity without consuming tokens
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
provider.DeepSeek(apiKey)        // DeepSeek Chat/Reasoner
provider.Gemini(apiKey)          // Google Gemini
provider.GLM(apiKey)             // Zhipu AI GLM
provider.Kimi(apiKey)            // Moonshot AI Kimi
provider.Qwen(apiKey)            // Alibaba Qwen (DashScope)
provider.MiniMax(apiKey)         // MiniMax
provider.Ollama("llama3")        // Local — Ollama
provider.VLLM("mistral")        // Local — vLLM
provider.ClaudeCLI()             // Claude CLI (exec-based, uses local claude binary)
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

## Health Check

Verify provider connectivity without consuming tokens:

```go
status := client.Ping(ctx)
if status.OK {
    fmt.Printf("Provider %s is up (latency: %v, models: %d)\n",
        status.Provider, status.Latency, status.Models)
} else {
    fmt.Printf("Provider %s is down: %v\n", status.Provider, status.Error)
}
```

`Ping()` tries `ListModels` first (free), falls back to a minimal completion if the provider doesn't support model listing.

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

## PDF/Document Input

Send PDFs and documents to models that support them (Anthropic):

```go
pdfBytes, _ := os.ReadFile("document.pdf")
resp, _ := client.Chat(ctx, []allm.Message{
    {Role: allm.RoleUser, Content: "Summarize this document",
     Documents: []allm.Document{{MimeType: "application/pdf", Data: pdfBytes, Name: "document.pdf"}}},
})
```

## Audio (TTS/STT)

Text-to-speech and speech-to-text with OpenAI:

```go
// Text-to-speech
resp, _ := client.Speak(ctx, &allm.SpeechRequest{
    Input:  "Hello, this is a test",
    Model:  "tts-1-hd",
    Voice:  "alloy",
    Format: "mp3",
})
os.WriteFile("output.mp3", resp.Audio, 0644)

// Speech-to-text (Whisper)
audioBytes, _ := os.ReadFile("audio.mp3")
resp, _ := client.Transcribe(ctx, &allm.TranscribeRequest{
    Audio:    audioBytes,
    Model:    "whisper-1",
    Language: "en",
})
fmt.Println("Transcription:", resp.Text)
```

## Citations

Extract citations from model responses (Anthropic):

```go
resp, _ := client.Complete(ctx, "Tell me about climate change with sources")
for _, cite := range resp.Citations {
    fmt.Printf("Citation: %s - %s\n", cite.Title, cite.URL)
}
```

## Streaming with Token Usage

Get token counts in streaming responses:

```go
for chunk := range client.Stream(ctx, messages) {
    if chunk.Error != nil {
        log.Fatal(chunk.Error)
    }
    fmt.Print(chunk.Content)
    if chunk.Done && chunk.Usage != nil {
        fmt.Printf("\nTokens: input=%d output=%d\n", chunk.Usage.InputTokens, chunk.Usage.OutputTokens)
    }
}
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

## Log Probabilities

Get per-token log probabilities for confidence scoring (OpenAI/compatible):

```go
client := allm.New(p, allm.WithLogProbs(5)) // Enable with top-5 alternatives per token

resp, _ := client.Complete(ctx, "The capital of France is")
for _, tokenLogProb := range resp.LogProbs {
    fmt.Printf("Token: %s, LogProb: %.4f\n", tokenLogProb.Token, tokenLogProb.LogProb)
    for _, alt := range tokenLogProb.TopLogProbs {
        fmt.Printf("  Alt: %s, LogProb: %.4f\n", alt.Token, alt.LogProb)
    }
}
```

Or set per-request:

```go
req := &allm.Request{
    Messages:    messages,
    LogProbs:    true,
    TopLogProbs: 10, // 0-20
}
resp, _ := client.Chat(ctx, req.Messages)
```

## Reproducible Outputs

Use seed for deterministic outputs (OpenAI/compatible):

```go
var seed int64 = 42
client := allm.New(p, allm.WithSeed(seed))

resp, _ := client.Complete(ctx, "Generate a random story")
fmt.Println("System fingerprint:", resp.SystemFingerprint) // Tracks system changes
```

Or set per-request:

```go
seed := int64(123)
req := &allm.Request{
    Messages: messages,
    Seed:     &seed,
}
```

## Parallel Tool Calls

Control parallel tool calling behavior (OpenAI/compatible):

```go
parallelToolCalls := false
req := &allm.Request{
    Messages:          messages,
    Tools:             tools,
    ParallelToolCalls: &parallelToolCalls, // Force sequential tool calls
}
```

## Predicted Output

Efficient editing with predicted content (OpenAI):

```go
req := &allm.Request{
    Messages: []allm.Message{
        {Role: allm.RoleUser, Content: "Fix typos in this text: ..."},
    },
    Prediction: &allm.PredictedOutput{
        Content: originalText, // The text you expect the model to output
    },
}
// Model only generates the diff, saving tokens and latency
```

## Request Tracking

Capture provider request IDs for debugging:

```go
resp, _ := client.Complete(ctx, "Hello")
fmt.Println("Request ID:", resp.RequestID)
// Anthropic: message ID (e.g., "msg_abc123")
// OpenAI: x-request-id header (if exposed by SDK)
```

## Thinking Streaming

Stream thinking/reasoning content separately (Anthropic):

```go
client := allm.New(provider.Anthropic(""), allm.WithThinking(5000))

for chunk := range client.Stream(ctx, messages) {
    if chunk.Thinking != "" {
        fmt.Print("[Thinking] ", chunk.Thinking)
    }
    if chunk.Content != "" {
        fmt.Print(chunk.Content)
    }
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
allm.WithLogProbs(5)                         // enable log probabilities with top-5 alternatives
allm.WithSeed(42)                            // reproducible outputs with seed
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

Sentinel errors: `ErrRateLimited`, `ErrServerError`, `ErrOverloaded`, `ErrTimeout`, `ErrInputTooLong`, `ErrEmptyInput`, `ErrNoProvider`, `ErrEmptyResponse`, `ErrCanceled`, `ErrProvider`, `ErrNotSupported`.

`ErrRateLimited` (429), `ErrServerError` (5xx), `ErrOverloaded` (529), `ErrTimeout`, and `ErrEmptyResponse` are automatically retried when `WithMaxRetries()` is set.

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
