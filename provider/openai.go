package provider

import (
	"context"
	"encoding/base64"
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

	// Create client once for connection reuse
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

	messages := p.convertMessages(req.Messages)

	// Resolve model: request > provider default
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

	if req.TopP > 0 {
		params.TopP = openai.Float(req.TopP)
	}

	// Stop sequences handled via SDK's native interface if supported

	completion, err := p.client.Chat.Completions.New(ctx, params)
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
		Provider:     "openai",
		Model:        model,
		InputTokens:  int(completion.Usage.PromptTokens),
		OutputTokens: int(completion.Usage.CompletionTokens),
		Latency:      time.Since(start),
		FinishReason: finishReason,
	}, nil
}

// Models returns available models from OpenAI.
func (p *OpenAIProvider) Models(ctx context.Context) ([]allm.Model, error) {
	page, err := p.client.Models.List(ctx)
	if err != nil {
		return nil, err
	}

	var models []allm.Model
	for _, m := range page.Data {
		models = append(models, allm.Model{
			ID:       m.ID,
			Name:     m.ID,
			Provider: "openai",
		})
	}
	return models, nil
}

// Stream sends a streaming request.
// Note: Streaming falls back to non-streaming for simplicity.
// For true streaming, use the SDK directly.
func (p *OpenAIProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
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

func (p *OpenAIProvider) buildClient() openai.Client {
	opts := []option.RequestOption{
		option.WithAPIKey(p.apiKey),
	}
	if p.baseURL != "" {
		opts = append(opts, option.WithBaseURL(p.baseURL))
	}
	return openai.NewClient(opts...)
}

func (p *OpenAIProvider) convertMessages(msgs []allm.Message) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion

	for _, m := range msgs {
		if len(m.Images) > 0 {
			var parts []openai.ChatCompletionContentPartUnionParam

			if m.Content != "" {
				parts = append(parts, openai.TextContentPart(m.Content))
			}

			for _, img := range m.Images {
				data := base64.StdEncoding.EncodeToString(img.Data)
				url := "data:" + img.MimeType + ";base64," + data
				parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
					URL: url,
				}))
			}

			switch m.Role {
			case allm.RoleUser:
				messages = append(messages, openai.UserMessage(parts))
			}
		} else {
			switch m.Role {
			case allm.RoleSystem:
				messages = append(messages, openai.SystemMessage(m.Content))
			case allm.RoleUser:
				messages = append(messages, openai.UserMessage(m.Content))
			case allm.RoleAssistant:
				messages = append(messages, openai.AssistantMessage(m.Content))
			}
		}
	}

	return messages
}
