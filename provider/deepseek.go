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

	client := p.newClient()
	messages := convertToOpenAI(req.Messages)

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
		Provider:     "deepseek",
		Model:        p.model,
		InputTokens:  int(completion.Usage.PromptTokens),
		OutputTokens: int(completion.Usage.CompletionTokens),
		Latency:      time.Since(start),
		FinishReason: finishReason,
	}, nil
}

// Models returns available models from DeepSeek.
func (p *DeepSeekProvider) Models(ctx context.Context) ([]allm.Model, error) {
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
			Provider: "deepseek",
		})
	}
	return models, nil
}

// Stream sends a streaming request.
// Note: Streaming falls back to non-streaming for simplicity.
func (p *DeepSeekProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
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

func (p *DeepSeekProvider) newClient() openai.Client {
	return openai.NewClient(
		option.WithAPIKey(p.apiKey),
		option.WithBaseURL("https://api.deepseek.com/v1"),
	)
}

// convertToOpenAI is a helper shared by OpenAI-compatible providers.
func convertToOpenAI(msgs []allm.Message) []openai.ChatCompletionMessageParamUnion {
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
