package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// wrapOpenAIError wraps OpenAI-compatible API errors with allm sentinel errors.
func wrapOpenAIError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr *openai.Error
	if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusTooManyRequests {
		return fmt.Errorf("%w: %w", allm.ErrRateLimited, err)
	}
	return err
}


// DeepSeekProvider implements allm.Provider for DeepSeek models.
type DeepSeekProvider struct {
	apiKey      string
	model       string
	maxTokens   int
	temperature float64
	client      openai.Client
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

	p.client = openai.NewClient(
		option.WithAPIKey(p.apiKey),
		option.WithBaseURL("https://api.deepseek.com/v1"),
	)

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

// buildChatParams builds ChatCompletionNewParams from an allm.Request.
func (p *DeepSeekProvider) buildChatParams(req *allm.Request) openai.ChatCompletionNewParams {
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
func (p *DeepSeekProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
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
		Provider:     "deepseek",
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

// Models returns available models from DeepSeek.
func (p *DeepSeekProvider) Models(ctx context.Context) ([]allm.Model, error) {
	page, err := p.client.Models.List(ctx)
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

// Stream sends a real streaming request using the OpenAI-compatible SDK.
func (p *DeepSeekProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
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

// convertToOpenAI is a helper shared by OpenAI-compatible providers (DeepSeek, GLM, Local).
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
