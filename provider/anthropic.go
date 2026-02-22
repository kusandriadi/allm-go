// Package provider implements LLM providers for allm-go.
package provider

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/kusandriadi/allm-go"
)

// wrapAnthropicError wraps Anthropic API errors with allm sentinel errors.
// Returns ErrRateLimited for 429 status codes, otherwise returns the original error.
func wrapAnthropicError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusTooManyRequests {
		return fmt.Errorf("%w: %w", allm.ErrRateLimited, err)
	}
	return err
}

// AnthropicProvider implements allm.Provider for Anthropic Claude.
type AnthropicProvider struct {
	apiKey      string
	authToken   string // OAuth token (Claude Pro/Max subscription)
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

// WithAnthropicAuthToken sets an OAuth token for authentication (Claude Pro/Max subscription).
// When set, uses Authorization: Bearer header instead of X-Api-Key.
// Takes precedence over API key.
func WithAnthropicAuthToken(token string) AnthropicOption {
	return func(p *AnthropicProvider) {
		p.authToken = token
	}
}

// Anthropic creates a new Anthropic provider.
//
// Authentication (in order of precedence):
//  1. WithAnthropicAuthToken option or ANTHROPIC_AUTH_TOKEN env → OAuth Bearer token (Claude Pro/Max)
//  2. apiKey parameter or ANTHROPIC_API_KEY env → API key (direct Anthropic API)
func Anthropic(apiKey string, opts ...AnthropicOption) *AnthropicProvider {
	p := &AnthropicProvider{
		model:     "claude-sonnet-4-20250514",
		maxTokens: 4096,
	}

	for _, opt := range opts {
		opt(p)
	}

	// Resolve auth: authToken takes precedence over apiKey
	if p.authToken == "" {
		p.authToken = os.Getenv("ANTHROPIC_AUTH_TOKEN")
	}
	if p.authToken == "" && apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	p.apiKey = apiKey

	// Validate custom base URL for security (SSRF prevention)
	if p.baseURL != "" {
		if err := validateBaseURLProvider(p.baseURL, false); err != nil {
			panic(fmt.Sprintf("anthropic: %v", err))
		}
	}

	// Create client once for connection reuse
	var clientOpts []option.RequestOption
	if p.authToken != "" {
		clientOpts = append(clientOpts, option.WithAuthToken(p.authToken))
	} else {
		clientOpts = append(clientOpts, option.WithAPIKey(p.apiKey))
	}
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

// Available returns true if authentication is configured (API key or OAuth token).
func (p *AnthropicProvider) Available() bool {
	return p.apiKey != "" || p.authToken != ""
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

		// Handle tool result messages — sent as user messages with tool_result blocks
		if m.Role == allm.RoleTool {
			var parts []anthropic.ContentBlockParamUnion
			for _, tr := range m.ToolResults {
				parts = append(parts, anthropic.NewToolResultBlock(tr.ToolCallID, tr.Content, tr.IsError))
			}
			messages = append(messages, anthropic.NewUserMessage(parts...))
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

		// Handle assistant messages with tool calls
		for _, tc := range m.ToolCalls {
			var input map[string]any
			if err := json.Unmarshal(tc.Arguments, &input); err != nil {
				return anthropic.MessageNewParams{}, fmt.Errorf("allm: invalid tool call arguments for %q: %w", tc.Name, err)
			}
			parts = append(parts, anthropic.ContentBlockParamUnion{
				OfToolUse: &anthropic.ToolUseBlockParam{
					ID:    tc.ID,
					Name:  tc.Name,
					Input: input,
				},
			})
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

	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			params.Tools = append(params.Tools, anthropic.ToolUnionParam{
				OfTool: &anthropic.ToolParam{
					Name:        t.Name,
					Description: anthropic.String(t.Description),
					InputSchema: anthropic.ToolInputSchemaParam{
						Properties: t.Parameters["properties"],
						Required:   toStringSlice(t.Parameters["required"]),
					},
				},
			})
		}
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
		return nil, wrapAnthropicError(err)
	}

	if len(message.Content) == 0 {
		return nil, allm.ErrEmptyResponse
	}

	resp := &allm.Response{
		Provider:     "anthropic",
		Model:        model,
		InputTokens:  int(message.Usage.InputTokens),
		OutputTokens: int(message.Usage.OutputTokens),
		Latency:      time.Since(start),
		FinishReason: string(message.StopReason),
	}

	for _, block := range message.Content {
		switch block.Type {
		case "text":
			resp.Content += block.Text
		case "tool_use":
			resp.ToolCalls = append(resp.ToolCalls, allm.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: json.RawMessage(block.Input),
			})
		}
	}

	return resp, nil
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
