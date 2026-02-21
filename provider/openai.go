package provider

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// OpenAIProvider implements allm.Provider for OpenAI GPT models.
type OpenAIProvider struct {
	apiKey      string
	model       string
	maxTokens   int
	temperature float64
	baseURL     string
	client      openai.Client
}

// OpenAIOption configures the OpenAI provider.
type OpenAIOption func(*OpenAIProvider)

// WithOpenAIModel sets the model.
func WithOpenAIModel(model string) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.model = model
	}
}

// WithOpenAIMaxTokens sets max output tokens.
func WithOpenAIMaxTokens(n int) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.maxTokens = n
	}
}

// WithOpenAITemperature sets the temperature.
func WithOpenAITemperature(t float64) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.temperature = t
	}
}

// WithOpenAIBaseURL sets a custom base URL (for Azure, proxies).
func WithOpenAIBaseURL(url string) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.baseURL = url
	}
}

// OpenAI creates a new OpenAI provider.
// If apiKey is empty, it reads from OPENAI_API_KEY environment variable.
func OpenAI(apiKey string, opts ...OpenAIOption) *OpenAIProvider {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	p := &OpenAIProvider{
		apiKey:    apiKey,
		model:     "gpt-4o",
		maxTokens: 4096,
	}

	for _, opt := range opts {
		opt(p)
	}

	p.client = p.buildClient()

	return p
}

// Name returns the provider name.
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// Available returns true if the API key is set.
func (p *OpenAIProvider) Available() bool {
	return p.apiKey != ""
}

// Complete sends a completion request.
func (p *OpenAIProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()

	messages := convertToOpenAI(req.Messages)
	params := openaiChatParams(messages, p.model, p.maxTokens, p.temperature, req)
	model := resolveModel(req.Model, p.model)

	completion, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, wrapOpenAIError(err)
	}

	return openaiCompleteResponse(completion, "openai", model, start)
}

// Models returns available models from OpenAI.
func (p *OpenAIProvider) Models(ctx context.Context) ([]allm.Model, error) {
	return openaiListModels(ctx, p.client, "openai")
}

// Embed generates embeddings using the OpenAI Embeddings API.
func (p *OpenAIProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	return openaiEmbed(ctx, p.client, req, "text-embedding-3-small", "openai")
}

// Stream sends a real streaming request using the OpenAI SDK.
func (p *OpenAIProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		messages := convertToOpenAI(req.Messages)
		params := openaiChatParams(messages, p.model, p.maxTokens, p.temperature, req)

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)
		openaiStreamLoop(stream, out)
	}()

	return out
}

func (p *OpenAIProvider) buildClient() openai.Client {
	// Validate custom base URL for security (SSRF prevention)
	if p.baseURL != "" {
		if err := validateBaseURLProvider(p.baseURL, false); err != nil {
			panic(fmt.Sprintf("openai: %v", err))
		}
	}

	opts := []option.RequestOption{
		option.WithAPIKey(p.apiKey),
	}
	if p.baseURL != "" {
		opts = append(opts, option.WithBaseURL(p.baseURL))
	}
	return openai.NewClient(opts...)
}
