package provider

import (
	"context"
	"os"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// DeepSeekProvider implements allm.Provider for DeepSeek models.
type DeepSeekProvider struct {
	apiKey      string
	model       string
	maxTokens   int
	temperature float64
	client      openai.Client
}

// DeepSeekOption configures the DeepSeek provider.
type DeepSeekOption func(*DeepSeekProvider)

// WithDeepSeekModel sets the model.
func WithDeepSeekModel(model string) DeepSeekOption {
	return func(p *DeepSeekProvider) {
		p.model = model
	}
}

// WithDeepSeekMaxTokens sets max output tokens.
func WithDeepSeekMaxTokens(n int) DeepSeekOption {
	return func(p *DeepSeekProvider) {
		p.maxTokens = n
	}
}

// WithDeepSeekTemperature sets the temperature.
func WithDeepSeekTemperature(t float64) DeepSeekOption {
	return func(p *DeepSeekProvider) {
		p.temperature = t
	}
}

// DeepSeek creates a new DeepSeek provider.
// If apiKey is empty, it reads from DEEPSEEK_API_KEY environment variable.
func DeepSeek(apiKey string, opts ...DeepSeekOption) *DeepSeekProvider {
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}

	p := &DeepSeekProvider{
		apiKey:    apiKey,
		model:     "deepseek-chat",
		maxTokens: 4096,
	}

	for _, opt := range opts {
		opt(p)
	}

	p.client = openai.NewClient(
		option.WithAPIKey(p.apiKey),
		option.WithBaseURL("https://api.deepseek.com/v1"),
	)

	return p
}

// Name returns the provider name.
func (p *DeepSeekProvider) Name() string {
	return "deepseek"
}

// Available returns true if the API key is set.
func (p *DeepSeekProvider) Available() bool {
	return p.apiKey != ""
}

// Complete sends a completion request.
func (p *DeepSeekProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
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

	return openaiCompleteResponse(completion, "deepseek", model, start)
}

// Models returns available models from DeepSeek.
func (p *DeepSeekProvider) Models(ctx context.Context) ([]allm.Model, error) {
	return openaiListModels(ctx, p.client, "deepseek")
}

// Stream sends a real streaming request using the OpenAI-compatible SDK.
func (p *DeepSeekProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
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
