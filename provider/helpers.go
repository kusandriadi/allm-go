package provider

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared" // ResponseFormat types
)

// validateBaseURLProvider checks if a base URL is safe from SSRF attacks.
// Uses net.ParseIP for robust IP validation (handles hex, octal, IPv6-mapped IPv4, etc).
// Note: DNS rebinding (domain resolving to internal IP) is not covered here;
// for production use with untrusted input, use a custom HTTP transport with IP validation.
func validateBaseURLProvider(baseURL string, allowLocal bool) error {
	if baseURL == "" {
		return fmt.Errorf("base URL cannot be empty")
	}

	u, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("invalid base URL: %w", err)
	}

	// Only allow http and https schemes
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("base URL must use http or https scheme, got: %s", u.Scheme)
	}

	// Reject userinfo in URL (e.g., http://evil@127.0.0.1/)
	if u.User != nil {
		return fmt.Errorf("base URL must not contain userinfo")
	}

	// Extract hostname for validation
	hostname := u.Hostname()
	if hostname == "" {
		return fmt.Errorf("base URL must include a hostname")
	}

	if !allowLocal {
		if isLocalOrPrivate(hostname) {
			return fmt.Errorf("base URL cannot point to localhost or private network (use Local provider for local servers)")
		}
	}

	return nil
}

// isLocalOrPrivate returns true if the hostname resolves to a loopback, private,
// link-local, or unspecified address. Uses net.ParseIP for robust handling of all
// IP representations (hex, octal, decimal, IPv6-mapped IPv4, bracketed, etc).
func isLocalOrPrivate(hostname string) bool {
	hostname = strings.ToLower(hostname)

	// Check well-known hostnames
	if hostname == "localhost" {
		return true
	}

	// Parse as IP address (handles all formats: dotted, hex, octal, IPv6, etc)
	ip := net.ParseIP(hostname)
	if ip == nil {
		// Not an IP — could be a domain name. We can't block DNS rebinding here,
		// but we block known dangerous hostnames.
		return false
	}

	return ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsUnspecified()
}

// sanitizeProviderError returns a safe error string for debug logging.
// Strips potential API keys, auth tokens, and sensitive URL components.
func sanitizeProviderError(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()
	// Check for patterns that indicate API key leakage
	lower := strings.ToLower(msg)
	sensitivePatterns := []string{
		"sk-ant-", "sk-", "gsk_", "api_key", "apikey",
		"bearer ", "token=", "key=", "authorization:",
	}
	for _, p := range sensitivePatterns {
		if strings.Contains(lower, p) {
			return "(error redacted: may contain credentials)"
		}
	}
	return msg
}

// wrapOpenAIError wraps OpenAI-compatible API errors with allm sentinel errors.
// Maps HTTP status codes to allm errors:
//   - 429 → ErrRateLimited
//   - 529 → ErrOverloaded
//   - 500-599 → ErrServerError
func wrapOpenAIError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr *openai.Error
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

// convertToOpenAI converts allm messages to OpenAI SDK format with image support.
// Shared by all OpenAI-compatible providers (OpenAI, DeepSeek, Gemini, GLM, etc).
func convertToOpenAI(msgs []allm.Message) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion

	for _, m := range msgs {
		// Handle tool result messages
		if m.Role == allm.RoleTool {
			for _, tr := range m.ToolResults {
				messages = append(messages, openai.ToolMessage(tr.Content, tr.ToolCallID))
			}
			continue
		}

		// Handle assistant messages with tool calls
		if len(m.ToolCalls) > 0 && m.Role == allm.RoleAssistant {
			msg := openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(m.Content),
				},
			}
			for _, tc := range m.ToolCalls {
				msg.ToolCalls = append(msg.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
					OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
						ID: tc.ID,
						Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name:      tc.Name,
							Arguments: string(tc.Arguments),
						},
					},
				})
			}
			messages = append(messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &msg})
			continue
		}

		// Handle messages with images (vision)
		if len(m.Images) > 0 && m.Role == allm.RoleUser {
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
			messages = append(messages, openai.UserMessage(parts))
			continue
		}

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

// convertToolsToOpenAI converts allm.Tool definitions to OpenAI SDK format.
func convertToolsToOpenAI(tools []allm.Tool) []openai.ChatCompletionToolUnionParam {
	var result []openai.ChatCompletionToolUnionParam
	for _, t := range tools {
		result = append(result, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        t.Name,
			Description: openai.String(t.Description),
			Parameters:  shared.FunctionParameters(t.Parameters),
		}))
	}
	return result
}

// toStringSlice converts an interface to []string for JSON schema required fields.
func toStringSlice(v any) []string {
	if v == nil {
		return nil
	}
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	result := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.(string); ok {
			result = append(result, s)
		}
	}
	return result
}

// openaiChatParams builds ChatCompletionNewParams from an allm.Request using provider defaults.
// defaultModel, defaultMaxTokens, and defaultTemp are the provider-level defaults;
// values in req take precedence when set.
func openaiChatParams(
	msgs []openai.ChatCompletionMessageParamUnion,
	defaultModel string,
	defaultMaxTokens int,
	defaultTemp float64,
	req *allm.Request,
) openai.ChatCompletionNewParams {
	m := defaultModel
	if req.Model != "" {
		m = req.Model
	}

	mt := int64(defaultMaxTokens)
	if req.MaxTokens > 0 {
		mt = int64(req.MaxTokens)
	}

	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(m),
		Messages:  msgs,
		MaxTokens: openai.Int(mt),
	}

	t := defaultTemp
	if req.Temperature > 0 {
		t = req.Temperature
	}
	if t > 0 {
		params.Temperature = openai.Float(t)
	}

	if req.TopP > 0 {
		params.TopP = openai.Float(req.TopP)
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

	// Structured output: response_format
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case allm.ResponseFormatJSON:
			params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
			}
		case allm.ResponseFormatJSONSchema:
			params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:   req.ResponseFormat.Name,
						Schema: req.ResponseFormat.Schema,
					},
				},
			}
		}
	}

	return params
}

// resolveModel returns req.Model if set, otherwise the provider default.
func resolveModel(reqModel, defaultModel string) string {
	if reqModel != "" {
		return reqModel
	}
	return defaultModel
}

// openaiCompleteResponse extracts an allm.Response from an OpenAI chat completion.
func openaiCompleteResponse(
	completion *openai.ChatCompletion,
	providerName string,
	model string,
	start time.Time,
) (*allm.Response, error) {
	if len(completion.Choices) == 0 {
		return nil, allm.ErrEmptyResponse
	}

	resp := &allm.Response{
		Content:      completion.Choices[0].Message.Content,
		Provider:     providerName,
		Model:        model,
		InputTokens:  int(completion.Usage.PromptTokens),
		OutputTokens: int(completion.Usage.CompletionTokens),
		Latency:      time.Since(start),
		FinishReason: string(completion.Choices[0].FinishReason),
	}

	// Extract reasoning content for DeepSeek Reasoner models
	// The SDK may expose this as part of the message in completion.Choices[0].Message
	// This is a placeholder - actual field name depends on SDK implementation
	// For now, we'll check if the response has extra fields (may need SDK update)

	for _, tc := range completion.Choices[0].Message.ToolCalls {
		resp.ToolCalls = append(resp.ToolCalls, allm.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}

	return resp, nil
}

// openaiStreamLoop reads from an OpenAI streaming response and forwards chunks to out.
func openaiStreamLoop(stream *ssestream.Stream[openai.ChatCompletionChunk], out chan<- allm.StreamChunk) {
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
}

// openaiListModels lists models from an OpenAI-compatible API.
func openaiListModels(ctx context.Context, client openai.Client, providerName string) ([]allm.Model, error) {
	page, err := client.Models.List(ctx)
	if err != nil {
		return nil, err
	}

	var models []allm.Model
	for _, m := range page.Data {
		models = append(models, allm.Model{
			ID:       m.ID,
			Name:     m.ID,
			Provider: providerName,
		})
	}
	return models, nil
}

// openaiEmbed generates embeddings using an OpenAI-compatible API.
func openaiEmbed(
	ctx context.Context,
	client openai.Client,
	req *allm.EmbedRequest,
	defaultModel string,
	providerName string,
) (*allm.EmbedResponse, error) {
	start := time.Now()

	model := defaultModel
	if req.Model != "" {
		model = req.Model
	}

	resp, err := client.Embeddings.New(ctx, openai.EmbeddingNewParams{
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
		Provider:    providerName,
		InputTokens: int(resp.Usage.TotalTokens),
		Latency:     time.Since(start),
	}, nil
}
