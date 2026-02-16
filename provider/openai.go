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

// buildChatParams builds ChatCompletionNewParams from an allm.Request.
// OpenAI has its own buildChatParams because it handles image messages.
func (p *OpenAIProvider) buildChatParams(req *allm.Request) openai.ChatCompletionNewParams {
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

	return params
}

// Complete sends a completion request.
func (p *OpenAIProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
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

	return openaiCompleteResponse(completion, "openai", model, start)
}

// Models returns available models from OpenAI.
func (p *OpenAIProvider) Models(ctx context.Context) ([]allm.Model, error) {
	return openaiListModels(ctx, p.client, "openai")
}

// Embed generates embeddings using the OpenAI Embeddings API.
func (p *OpenAIProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	return openaiEmbed(ctx, p.client, req, "text-embedding-3-small", "openai")
}

// Stream sends a real streaming request using the OpenAI SDK.
func (p *OpenAIProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		params := p.buildChatParams(req)

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)
		openaiStreamLoop(stream, out)
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

// convertMessages converts allm messages to OpenAI format with image support.
// OpenAI's convertMessages is separate from the shared convertToOpenAI because
// it handles vision (image) content parts.
func (p *OpenAIProvider) convertMessages(msgs []allm.Message) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion

	for _, m := range msgs {
		// Handle tool result messages
		if m.Role == allm.RoleTool {
			for _, tr := range m.ToolResults {
				messages = append(messages, openai.ToolMessage(tr.Content, tr.ToolCallID))
			}
			continue
		}

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
		} else if len(m.ToolCalls) > 0 && m.Role == allm.RoleAssistant {
			// Assistant message with tool calls
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
