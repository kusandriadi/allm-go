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
	// For local servers, we assume available if baseURL is set
	// Could add a health check here if needed
	return p.baseURL != ""
}

// Complete sends a completion request.
func (p *LocalProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()

	client := p.newClient()
	messages := p.convertMessages(req.Messages)

	maxTokens := int64(p.maxTokens)
	if req.MaxTokens > 0 {
		maxTokens = int64(req.MaxTokens)
	}

	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(p.model),
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

	completion, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	content := ""
	finishReason := ""
	if len(completion.Choices) > 0 {
		content = completion.Choices[0].Message.Content
		finishReason = string(completion.Choices[0].FinishReason)
	}

	return &allm.Response{
		Content:      content,
		Provider:     "local",
		Model:        p.model,
		InputTokens:  int(completion.Usage.PromptTokens),
		OutputTokens: int(completion.Usage.CompletionTokens),
		Latency:      time.Since(start),
		FinishReason: finishReason,
	}, nil
}

// Models returns available models from the local server.
func (p *LocalProvider) Models(ctx context.Context) ([]allm.Model, error) {
	client := p.newClient()

	page, err := client.Models.List(ctx)
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

// Stream sends a streaming request.
// Note: Streaming falls back to non-streaming for simplicity.
func (p *LocalProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		resp, err := p.Complete(ctx, req)
		if err != nil {
			out <- allm.StreamChunk{Error: err}
			return
		}

		out <- allm.StreamChunk{Content: resp.Content}
		out <- allm.StreamChunk{Done: true}
	}()

	return out
}

func (p *LocalProvider) newClient() openai.Client {
	opts := []option.RequestOption{
		option.WithBaseURL(p.baseURL),
	}
	if p.apiKey != "" {
		opts = append(opts, option.WithAPIKey(p.apiKey))
	} else {
		// Some servers require a dummy key
		opts = append(opts, option.WithAPIKey(os.Getenv("LOCAL_API_KEY")))
	}
	return openai.NewClient(opts...)
}

func (p *LocalProvider) convertMessages(msgs []allm.Message) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion

	for _, m := range msgs {
		switch m.Role {
		case allm.RoleSystem:
			messages = append(messages, openai.SystemMessage(m.Content))
		case allm.RoleUser:
			messages = append(messages, openai.UserMessage(m.Content))
		case allm.RoleAssistant:
			messages = append(messages, openai.AssistantMessage(m.Content))
		}
	}

	return messages
}
