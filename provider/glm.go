package provider

import (
	"context"
	"os"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// GLMProvider implements allm.Provider for Zhipu AI GLM models.
type GLMProvider struct {
	apiKey      string
	model       string
	maxTokens   int
	temperature float64
	client      openai.Client
}

// GLMOption configures the GLM provider.
type GLMOption func(*GLMProvider)

// WithGLMModel sets the model.
func WithGLMModel(model string) GLMOption {
	return func(p *GLMProvider) {
		p.model = model
	}
}

// WithGLMMaxTokens sets max output tokens.
func WithGLMMaxTokens(n int) GLMOption {
	return func(p *GLMProvider) {
		p.maxTokens = n
	}
}

// WithGLMTemperature sets the temperature.
func WithGLMTemperature(t float64) GLMOption {
	return func(p *GLMProvider) {
		p.temperature = t
	}
}

// GLM creates a new Zhipu AI GLM provider.
// If apiKey is empty, it reads from GLM_API_KEY environment variable.
func GLM(apiKey string, opts ...GLMOption) *GLMProvider {
	if apiKey == "" {
		apiKey = os.Getenv("GLM_API_KEY")
	}

	p := &GLMProvider{
		apiKey:    apiKey,
		model:     "glm-4-flash",
		maxTokens: 4096,
	}

	for _, opt := range opts {
		opt(p)
	}

	p.client = openai.NewClient(
		option.WithAPIKey(p.apiKey),
		option.WithBaseURL("https://open.bigmodel.cn/api/paas/v4/"),
	)

	return p
}

// Name returns the provider name.
func (p *GLMProvider) Name() string {
	return "glm"
}

// Available returns true if the API key is set.
func (p *GLMProvider) Available() bool {
	return p.apiKey != ""
}

// Complete sends a completion request.
func (p *GLMProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
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

	return openaiCompleteResponse(completion, "glm", model, start)
}

// Models returns available models from GLM.
func (p *GLMProvider) Models(ctx context.Context) ([]allm.Model, error) {
	return openaiListModels(ctx, p.client, "glm")
}

// Embed generates embeddings using the GLM Embeddings API.
func (p *GLMProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	return openaiEmbed(ctx, p.client, req, "embedding-3", "glm")
}

// Stream sends a real streaming request using the OpenAI-compatible SDK.
func (p *GLMProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
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
