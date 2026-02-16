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

// buildChatParams builds ChatCompletionNewParams from an allm.Request.
func (p *LocalProvider) buildChatParams(req *allm.Request) openai.ChatCompletionNewParams {
	messages := p.convertMessages(req.Messages)

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
func (p *LocalProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
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
		Provider:     "local",
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

// Models returns available models from the local server.
func (p *LocalProvider) Models(ctx context.Context) ([]allm.Model, error) {
	page, err := p.client.Models.List(ctx)
	if err != nil {
		return nil, err
	}

	var models []allm.Model
	for _, m := range page.Data {
		models = append(models, allm.Model{
			ID:       m.ID,
			Name:     m.ID,
			Provider: "local",
		})
	}
	return models, nil
}

// Embed generates embeddings using the local OpenAI-compatible API.
// Works with Ollama, vLLM, and other servers that support /v1/embeddings.
func (p *LocalProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	start := time.Now()

	model := p.model
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
		Provider:    "local",
		InputTokens: int(resp.Usage.TotalTokens),
		Latency:     time.Since(start),
	}, nil
}

// Stream sends a real streaming request using the OpenAI-compatible SDK.
func (p *LocalProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
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

func (p *LocalProvider) convertMessages(msgs []allm.Message) []openai.ChatCompletionMessageParamUnion {
	// Reuse the shared helper which handles tool messages
	return convertToOpenAI(msgs)
}
