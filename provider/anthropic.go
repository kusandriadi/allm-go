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
// Maps HTTP status codes to allm errors:
//   - 429 → ErrRateLimited
//   - 529 → ErrOverloaded (Anthropic-specific: API overloaded)
//   - 500-599 → ErrServerError
func wrapAnthropicError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		switch {
		case apiErr.StatusCode == http.StatusTooManyRequests:
			return fmt.Errorf("%w: %w", allm.ErrRateLimited, err)
		case apiErr.StatusCode == 529:
			return fmt.Errorf("%w: %w", allm.ErrOverloaded, err)
		case apiErr.StatusCode >= 500 && apiErr.StatusCode < 600:
			return fmt.Errorf("%w: %w", allm.ErrServerError, err)
		}
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
	logger      allm.Logger
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

// WithAnthropicLogger sets a logger for provider-level debug tracing.
func WithAnthropicLogger(logger allm.Logger) AnthropicOption {
	return func(p *AnthropicProvider) {
		p.logger = logger
	}
}

// Anthropic creates a new Anthropic provider.
//
// Authentication (in order of precedence):
//  1. WithAnthropicAuthToken option or CLAUDE_CODE_OAUTH_TOKEN env → OAuth Bearer token (Claude Pro/Max)
//  2. apiKey parameter or ANTHROPIC_API_KEY env → API key (direct Anthropic API)
func Anthropic(apiKey string, opts ...AnthropicOption) *AnthropicProvider {
	p := &AnthropicProvider{
		model:     "claude-sonnet-4-6",
		maxTokens: 4096,
	}

	for _, opt := range opts {
		opt(p)
	}

	// Resolve auth: authToken takes precedence over apiKey
	if p.authToken == "" {
		p.authToken = os.Getenv("CLAUDE_CODE_OAUTH_TOKEN")
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
			block := anthropic.TextBlockParam{Text: m.Content}
			if m.CacheControl != nil {
				block.CacheControl = anthropic.NewCacheControlEphemeralParam()
			}
			systemBlocks = append(systemBlocks, block)
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
			textBlock := anthropic.NewTextBlock(m.Content)
			if m.CacheControl != nil {
				textBlock.OfText.CacheControl = anthropic.NewCacheControlEphemeralParam()
			}
			parts = append(parts, textBlock)
		}

		for _, img := range m.Images {
			data := base64.StdEncoding.EncodeToString(img.Data)
			parts = append(parts, anthropic.NewImageBlockBase64(img.MimeType, data))
		}

		for _, doc := range m.Documents {
			data := base64.StdEncoding.EncodeToString(doc.Data)
			// Anthropic supports PDF documents natively via document blocks
			parts = append(parts, anthropic.ContentBlockParamUnion{
				OfDocument: &anthropic.DocumentBlockParam{
					Source: anthropic.DocumentBlockParamSourceUnion{
						OfBase64: &anthropic.Base64PDFSourceParam{
							Data: data,
						},
					},
				},
			})
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
	// Note: Anthropic requires that temperature must NOT be set when thinking is enabled
	if temp > 0 && req.Thinking == nil {
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

	// Computer use tool
	// TODO: Implement computer use tool support via Beta API
	// Computer use requires using BetaToolComputerUse20241022Param in the Beta API
	// For now, ignore ComputerUse field (no error, just not implemented)
	_ = req.ComputerUse

	// Extended thinking / reasoning
	if req.Thinking != nil && req.Thinking.BudgetTokens > 0 {
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(req.Thinking.BudgetTokens))
	}

	return params, nil
}

// Complete sends a completion request.
func (p *AnthropicProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	if p.logger != nil {
		p.logger.Debug("provider complete",
			"provider", "anthropic",
			"model", model,
			"messages", len(req.Messages),
		)
	}

	params, err := p.buildParams(req)
	if err != nil {
		return nil, err
	}

	message, err := p.client.Messages.New(ctx, params)
	if err != nil {
		if p.logger != nil {
			p.logger.Debug("provider complete failed",
				"provider", "anthropic",
				"model", model,
				"latency", time.Since(start),
				"error", sanitizeProviderError(err),
			)
		}
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
		RequestID:    message.ID, // Anthropic message ID for debugging
	}

	// Extract cache token usage if available
	if message.Usage.CacheReadInputTokens > 0 {
		resp.CacheReadTokens = int(message.Usage.CacheReadInputTokens)
	}
	if message.Usage.CacheCreationInputTokens > 0 {
		resp.CacheWriteTokens = int(message.Usage.CacheCreationInputTokens)
	}

	for _, block := range message.Content {
		switch block.Type {
		case "text":
			resp.Content += block.Text
		case "thinking":
			// Extract thinking content
			resp.Thinking += block.Thinking
			// Thinking tokens are tracked separately in the SDK
		case "tool_use":
			resp.ToolCalls = append(resp.ToolCalls, allm.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: json.RawMessage(block.Input),
			})
			// TODO: Parse citation blocks when SDK exposes citation type
			// case "citation":
			//     resp.Citations = append(resp.Citations, allm.Citation{...})
		}
	}

	if p.logger != nil {
		logArgs := []any{
			"provider", "anthropic",
			"model", model,
			"latency", resp.Latency,
			"input_tokens", resp.InputTokens,
			"output_tokens", resp.OutputTokens,
			"finish_reason", resp.FinishReason,
		}
		if len(resp.ToolCalls) > 0 {
			logArgs = append(logArgs, "tool_calls", len(resp.ToolCalls))
		}
		p.logger.Debug("provider complete done", logArgs...)
	}

	return resp, nil
}

// CountTokens estimates input tokens for a request using Anthropic's count_tokens endpoint.
func (p *AnthropicProvider) CountTokens(ctx context.Context, req *allm.Request) (*allm.TokenCount, error) {
	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	if p.logger != nil {
		p.logger.Debug("provider count tokens",
			"provider", "anthropic",
			"model", model,
			"messages", len(req.Messages),
		)
	}

	params, err := p.buildParams(req)
	if err != nil {
		return nil, err
	}

	countParams := anthropic.MessageCountTokensParams{
		Model:    anthropic.Model(model),
		Messages: params.Messages,
	}

	// Pass system prompt for accurate counting
	if len(params.System) > 0 {
		countParams.System = anthropic.MessageCountTokensParamsSystemUnion{
			OfTextBlockArray: params.System,
		}
	}

	// Pass thinking config for accurate counting
	if req.Thinking != nil && req.Thinking.BudgetTokens > 0 {
		countParams.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(req.Thinking.BudgetTokens))
	}

	result, err := p.client.Messages.CountTokens(ctx, countParams)
	if err != nil {
		if p.logger != nil {
			p.logger.Debug("provider count tokens failed",
				"provider", "anthropic",
				"model", model,
				"error", sanitizeProviderError(err),
			)
		}
		return nil, err
	}

	if p.logger != nil {
		p.logger.Debug("provider count tokens done",
			"provider", "anthropic",
			"model", model,
			"input_tokens", result.InputTokens,
		)
	}

	return &allm.TokenCount{
		InputTokens: int(result.InputTokens),
		Provider:    "anthropic",
		Model:       model,
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
		model := allm.Model{
			ID:       m.ID,
			Name:     m.DisplayName,
			Provider: "anthropic",
		}
		// Populate metadata if available
		if m.MaxInputTokens > 0 {
			model.ContextWindow = int(m.MaxInputTokens)
		}
		if m.MaxTokens > 0 {
			model.MaxOutput = int(m.MaxTokens)
		}
		// Populate capabilities based on model type
		model.Capabilities = []string{"chat", "streaming"}
		// Most Claude models support vision and tools
		if m.ID != "" {
			model.Capabilities = append(model.Capabilities, "vision", "tools")
		}
		if !m.CreatedAt.IsZero() {
			model.CreatedAt = m.CreatedAt.Unix()
		}
		models = append(models, model)
	}
	return models, nil
}

// Stream sends a real streaming request using the Anthropic SDK.
func (p *AnthropicProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		model := p.model
		if req.Model != "" {
			model = req.Model
		}

		if p.logger != nil {
			p.logger.Debug("provider stream",
				"provider", "anthropic",
				"model", model,
				"messages", len(req.Messages),
			)
		}

		params, err := p.buildParams(req)
		if err != nil {
			out <- allm.StreamChunk{Error: err}
			return
		}

		stream := p.client.Messages.NewStreaming(ctx, params)
		defer stream.Close()

		var usage *allm.StreamUsage
		// Track content block types (index -> type) to handle thinking vs text
		blockTypes := make(map[int64]string)

		for stream.Next() {
			event := stream.Current()

			// content_block_start events tell us the block type
			if event.Type == "content_block_start" {
				blockTypes[event.Index] = event.ContentBlock.Type
			}

			// content_block_delta events contain text or thinking chunks
			if event.Type == "content_block_delta" {
				blockType := blockTypes[event.Index]
				if blockType == "thinking" && event.Delta.Thinking != "" {
					// Thinking content
					out <- allm.StreamChunk{Thinking: event.Delta.Thinking}
				} else if event.Delta.Text != "" {
					// Regular text content
					out <- allm.StreamChunk{Content: event.Delta.Text}
				}
			}

			// message_delta events contain usage information
			if event.Type == "message_delta" && event.Usage.OutputTokens > 0 {
				if usage == nil {
					usage = &allm.StreamUsage{}
				}
				usage.OutputTokens = int(event.Usage.OutputTokens)
			}
			// message_start events contain input token count
			if event.Type == "message_start" && event.Message.Usage.InputTokens > 0 {
				if usage == nil {
					usage = &allm.StreamUsage{}
				}
				usage.InputTokens = int(event.Message.Usage.InputTokens)
			}
		}

		if err := stream.Err(); err != nil {
			out <- allm.StreamChunk{Error: err}
			return
		}

		out <- allm.StreamChunk{Done: true, Usage: usage}
	}()

	return out
}
