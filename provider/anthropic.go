// Package provider implements LLM providers for allm-go.
package provider

import (
	"context"
	"encoding/base64"
	"os"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/kusandriadi/allm-go"
)

// AnthropicProvider implements allm.Provider for Anthropic Claude.
type AnthropicProvider struct {
	apiKey      string
	model       string
	maxTokens   int
	temperature float64
	baseURL     string
	client      anthropic.Client
}

// AnthropicOption configures the Anthropic provider.
type AnthropicOption func(*AnthropicProvider)

// WithAnthropicModel sets the model.
func WithAnthropicModel(model string) AnthropicOption {
	return func(p *AnthropicProvider) {
		p.model = model
	}
}

// WithAnthropicMaxTokens sets max output tokens.
func WithAnthropicMaxTokens(n int) AnthropicOption {
	return func(p *AnthropicProvider) {
		p.maxTokens = n
	}
}

// WithAnthropicTemperature sets the temperature.
func WithAnthropicTemperature(t float64) AnthropicOption {
	return func(p *AnthropicProvider) {
		p.temperature = t
	}
}

// WithAnthropicBaseURL sets a custom base URL (for proxies).
func WithAnthropicBaseURL(url string) AnthropicOption {
	return func(p *AnthropicProvider) {
		p.baseURL = url
	}
}

// Anthropic creates a new Anthropic provider.
// If apiKey is empty, it reads from ANTHROPIC_API_KEY environment variable.
func Anthropic(apiKey string, opts ...AnthropicOption) *AnthropicProvider {
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}

	p := &AnthropicProvider{
		apiKey:    apiKey,
		model:     "claude-sonnet-4-20250514",
		maxTokens: 4096,
	}

	for _, opt := range opts {
		opt(p)
	}

	// Create client once for connection reuse
	var clientOpts []option.RequestOption
	clientOpts = append(clientOpts, option.WithAPIKey(p.apiKey))
	if p.baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(p.baseURL))
	}
	p.client = anthropic.NewClient(clientOpts...)

	return p
}

// Name returns the provider name.
func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

// Available returns true if the API key is set.
func (p *AnthropicProvider) Available() bool {
	return p.apiKey != ""
}

// buildParams builds MessageNewParams from an allm.Request.
func (p *AnthropicProvider) buildParams(req *allm.Request) (anthropic.MessageNewParams, error) {
	var systemBlocks []anthropic.TextBlockParam
	var messages []anthropic.MessageParam

	for _, m := range req.Messages {
		if m.Role == allm.RoleSystem {
			systemBlocks = append(systemBlocks, anthropic.TextBlockParam{Text: m.Content})
			continue
		}

		var parts []anthropic.ContentBlockParamUnion

		if m.Content != "" {
			parts = append(parts, anthropic.NewTextBlock(m.Content))
		}

		for _, img := range m.Images {
			data := base64.StdEncoding.EncodeToString(img.Data)
			parts = append(parts, anthropic.NewImageBlockBase64(img.MimeType, data))
		}

		switch m.Role {
		case allm.RoleUser:
			messages = append(messages, anthropic.NewUserMessage(parts...))
		case allm.RoleAssistant:
			messages = append(messages, anthropic.NewAssistantMessage(parts...))
		}
	}

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	maxTokens := int64(p.maxTokens)
	if req.MaxTokens > 0 {
		maxTokens = int64(req.MaxTokens)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: maxTokens,
		Messages:  messages,
	}

	if len(systemBlocks) > 0 {
		params.System = systemBlocks
	}

	temp := p.temperature
	if req.Temperature > 0 {
		temp = req.Temperature
	}
	if temp > 0 {
		params.Temperature = anthropic.Float(temp)
	}

	if req.TopP > 0 {
		params.TopP = anthropic.Float(req.TopP)
	}

	if len(req.Stop) > 0 {
		params.StopSequences = req.Stop
	}

	return params, nil
}

// Complete sends a completion request.
func (p *AnthropicProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()

	params, err := p.buildParams(req)
	if err != nil {
		return nil, err
	}

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	message, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, err
	}

	content := ""
	if len(message.Content) > 0 {
		content = message.Content[0].Text
	}

	return &allm.Response{
		Content:      content,
		Provider:     "anthropic",
		Model:        model,
		InputTokens:  int(message.Usage.InputTokens),
		OutputTokens: int(message.Usage.OutputTokens),
		Latency:      time.Since(start),
		FinishReason: string(message.StopReason),
	}, nil
}

// Models returns available models from Anthropic.
func (p *AnthropicProvider) Models(ctx context.Context) ([]allm.Model, error) {
	page, err := p.client.Models.List(ctx, anthropic.ModelListParams{})
	if err != nil {
		return nil, err
	}

	var models []allm.Model
	for _, m := range page.Data {
		models = append(models, allm.Model{
			ID:       m.ID,
			Name:     m.DisplayName,
			Provider: "anthropic",
		})
	}
	return models, nil
}

// Stream sends a real streaming request using the Anthropic SDK.
func (p *AnthropicProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		params, err := p.buildParams(req)
		if err != nil {
			out <- allm.StreamChunk{Error: err}
			return
		}

		stream := p.client.Messages.NewStreaming(ctx, params)
		defer stream.Close()

		for stream.Next() {
			event := stream.Current()
			// content_block_delta events contain text chunks
			if event.Type == "content_block_delta" && event.Delta.Text != "" {
				out <- allm.StreamChunk{Content: event.Delta.Text}
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
