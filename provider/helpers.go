package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared"
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

// convertToOpenAI converts allm messages to OpenAI SDK format.
// Shared by all OpenAI-compatible providers (OpenAI, DeepSeek, GLM, Local).
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
func openaiChatParams(
	msgs []openai.ChatCompletionMessageParamUnion,
	model string, defaultModel string,
	maxTokens int, defaultMaxTokens int,
	temp float64, defaultTemp float64,
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
