package provider

import (
	"context"
	"os"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// LocalProvider implements allm.Provider for local/self-hosted models.
// Works with any OpenAI-compatible API:
//   - Ollama (http://localhost:11434/v1)
//   - vLLM (http://localhost:8000/v1)
//   - llama.cpp server
//   - LocalAI
//   - text-generation-webui
type LocalProvider struct {
	baseURL     string
	apiKey      string
	model       string
	maxTokens   int
	temperature float64
	client      openai.Client
}

// LocalOption configures the Local provider.
type LocalOption func(*LocalProvider)

// WithLocalModel sets the model name.
func WithLocalModel(model string) LocalOption {
	return func(p *LocalProvider) {
		p.model = model
	}
}

// WithLocalAPIKey sets an optional API key.
func WithLocalAPIKey(key string) LocalOption {
	return func(p *LocalProvider) {
		p.apiKey = key
	}
}

// WithLocalMaxTokens sets max output tokens.
func WithLocalMaxTokens(n int) LocalOption {
	return func(p *LocalProvider) {
		p.maxTokens = n
	}
}

// WithLocalTemperature sets the temperature.
func WithLocalTemperature(t float64) LocalOption {
	return func(p *LocalProvider) {
		p.temperature = t
	}
}

// Local creates a new Local provider for OpenAI-compatible servers.
// Default baseURL is http://localhost:11434/v1 (Ollama).
func Local(baseURL string, opts ...LocalOption) *LocalProvider {
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}

	p := &LocalProvider{
		baseURL:   baseURL,
		model:     "llama3",
		maxTokens: 4096,
	}

	for _, opt := range opts {
		opt(p)
	}

	p.client = p.buildClient()

	return p
}

// Ollama creates a Local provider configured for Ollama.
func Ollama(model string, opts ...LocalOption) *LocalProvider {
	opts = append([]LocalOption{WithLocalModel(model)}, opts...)
	return Local("http://localhost:11434/v1", opts...)
}

// VLLM creates a Local provider configured for vLLM.
func VLLM(model string, opts ...LocalOption) *LocalProvider {
	opts = append([]LocalOption{WithLocalModel(model)}, opts...)
	return Local("http://localhost:8000/v1", opts...)
}

// Name returns the provider name.
func (p *LocalProvider) Name() string {
	return "local"
}

// Available checks if the local server is reachable.
func (p *LocalProvider) Available() bool {
	return p.baseURL != ""
}

// Complete sends a completion request.
func (p *LocalProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()

	messages := convertToOpenAI(req.Messages)
	params := openaiChatParams(messages, req.Model, p.model, req.MaxTokens, p.maxTokens, req.Temperature, p.temperature, req)

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	completion, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, wrapOpenAIError(err)
	}

	return openaiCompleteResponse(completion, "local", model, start)
}

// Models returns available models from the local server.
func (p *LocalProvider) Models(ctx context.Context) ([]allm.Model, error) {
	return openaiListModels(ctx, p.client, "local")
}

// Embed generates embeddings using the local OpenAI-compatible API.
// Works with Ollama, vLLM, and other servers that support /v1/embeddings.
func (p *LocalProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	return openaiEmbed(ctx, p.client, req, p.model, "local")
}

// Stream sends a real streaming request using the OpenAI-compatible SDK.
func (p *LocalProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		messages := convertToOpenAI(req.Messages)
		params := openaiChatParams(messages, req.Model, p.model, req.MaxTokens, p.maxTokens, req.Temperature, p.temperature, req)

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)
		openaiStreamLoop(stream, out)
	}()

	return out
}

func (p *LocalProvider) buildClient() openai.Client {
	opts := []option.RequestOption{
		option.WithBaseURL(p.baseURL),
	}
	if p.apiKey != "" {
		opts = append(opts, option.WithAPIKey(p.apiKey))
	} else {
		opts = append(opts, option.WithAPIKey(os.Getenv("LOCAL_API_KEY")))
	}
	return openai.NewClient(opts...)
}
