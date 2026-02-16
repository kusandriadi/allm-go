package provider

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
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

	if len(completion.Choices) == 0 {
		return nil, allm.ErrEmptyResponse
	}

	resp := &allm.Response{
		Content:      completion.Choices[0].Message.Content,
		Provider:     "openai",
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

// Embed generates embeddings using the OpenAI Embeddings API.
func (p *OpenAIProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	start := time.Now()

	model := "text-embedding-3-small"
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
		Provider:    "openai",
		InputTokens: int(resp.Usage.TotalTokens),
		Latency:     time.Since(start),
	}, nil
}

// Stream sends a real streaming request using the OpenAI SDK.
func (p *OpenAIProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
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
