package provider

import (
	"context"
	"encoding/json"
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

// buildChatParams builds ChatCompletionNewParams from an allm.Request.
func (p *GLMProvider) buildChatParams(req *allm.Request) openai.ChatCompletionNewParams {
	messages := convertToOpenAI(req.Messages)

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	maxTokens := int64(p.maxTokens)
	if req.MaxTokens > 0 {
		maxTokens = int64(req.MaxTokens)
	}

	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(model),
		Messages:  messages,
		MaxTokens: openai.Int(maxTokens),
	}

	temp := p.temperature
	if req.Temperature > 0 {
		temp = req.Temperature
	}
	if temp > 0 {
		params.Temperature = openai.Float(temp)
	}

	if req.PresencePenalty != 0 {
		params.PresencePenalty = openai.Float(req.PresencePenalty)
	}

	if req.FrequencyPenalty != 0 {
		params.FrequencyPenalty = openai.Float(req.FrequencyPenalty)
	}

	if len(req.Stop) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: req.Stop}
	}

	if len(req.Tools) > 0 {
		params.Tools = convertToolsToOpenAI(req.Tools)
	}

	return params
}

// Complete sends a completion request.
func (p *GLMProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()

	params := p.buildChatParams(req)

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	completion, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, wrapOpenAIError(err)
	}

	if len(completion.Choices) == 0 {
		return nil, allm.ErrEmptyResponse
	}

	resp := &allm.Response{
		Content:      completion.Choices[0].Message.Content,
		Provider:     "glm",
		Model:        model,
		InputTokens:  int(completion.Usage.PromptTokens),
		OutputTokens: int(completion.Usage.CompletionTokens),
		Latency:      time.Since(start),
		FinishReason: string(completion.Choices[0].FinishReason),
	}

	for _, tc := range completion.Choices[0].Message.ToolCalls {
		resp.ToolCalls = append(resp.ToolCalls, allm.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}

	return resp, nil
}

// Models returns available models from GLM.
func (p *GLMProvider) Models(ctx context.Context) ([]allm.Model, error) {
	page, err := p.client.Models.List(ctx)
	if err != nil {
		return nil, err
	}

	var models []allm.Model
	for _, m := range page.Data {
		models = append(models, allm.Model{
			ID:       m.ID,
			Name:     m.ID,
			Provider: "glm",
		})
	}
	return models, nil
}

// Embed generates embeddings using the GLM Embeddings API.
func (p *GLMProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	start := time.Now()

	model := "embedding-3"
	if req.Model != "" {
		model = req.Model
	}

	resp, err := p.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: req.Input,
		},
		Model: openai.EmbeddingModel(model),
	})
	if err != nil {
		return nil, err
	}

	embeddings := make([][]float64, len(resp.Data))
	for i, e := range resp.Data {
		embeddings[i] = e.Embedding
	}

	return &allm.EmbedResponse{
		Embeddings:  embeddings,
		Model:       model,
		Provider:    "glm",
		InputTokens: int(resp.Usage.TotalTokens),
		Latency:     time.Since(start),
	}, nil
}

// Stream sends a real streaming request using the OpenAI-compatible SDK.
func (p *GLMProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		params := p.buildChatParams(req)

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)
		defer stream.Close()

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) > 0 {
				content := chunk.Choices[0].Delta.Content
				if content != "" {
					out <- allm.StreamChunk{Content: content}
				}
			}
		}

		if err := stream.Err(); err != nil {
			out <- allm.StreamChunk{Error: err}
			return
		}

		out <- allm.StreamChunk{Done: true}
	}()

	return out
}
