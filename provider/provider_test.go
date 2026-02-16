package provider

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/kusandriadi/allm-go"
	openai "github.com/openai/openai-go/v3"
)

// --- Anthropic ---

func TestAnthropicNew(t *testing.T) {
	p := Anthropic("")
	if p.Name() != "anthropic" {
		t.Errorf("expected 'anthropic', got %q", p.Name())
	}
	if p.model != "claude-sonnet-4-20250514" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.maxTokens != 4096 {
		t.Errorf("expected 4096, got %d", p.maxTokens)
	}
}

func TestAnthropicWithOptions(t *testing.T) {
	p := Anthropic("test-key",
		WithAnthropicModel("claude-opus-4"),
		WithAnthropicMaxTokens(8192),
		WithAnthropicTemperature(0.7),
		WithAnthropicBaseURL("https://custom.api"),
	)

	if p.apiKey != "test-key" {
		t.Error("API key not set")
	}
	if p.model != "claude-opus-4" {
		t.Error("model not set")
	}
	if p.maxTokens != 8192 {
		t.Error("maxTokens not set")
	}
	if p.temperature != 0.7 {
		t.Error("temperature not set")
	}
	if p.baseURL != "https://custom.api" {
		t.Error("baseURL not set")
	}
}

func TestAnthropicAvailable(t *testing.T) {
	p1 := Anthropic("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := Anthropic("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestAnthropicEnvKey(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	p := Anthropic("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

func TestAnthropicExplicitKeyOverridesEnv(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	p := Anthropic("explicit-key")
	if p.apiKey != "explicit-key" {
		t.Errorf("expected explicit key, got %q", p.apiKey)
	}
}

// --- OpenAI ---

func TestOpenAINew(t *testing.T) {
	p := OpenAI("")
	if p.Name() != "openai" {
		t.Errorf("expected 'openai', got %q", p.Name())
	}
	if p.model != "gpt-4o" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.maxTokens != 4096 {
		t.Errorf("expected 4096, got %d", p.maxTokens)
	}
}

func TestOpenAIWithOptions(t *testing.T) {
	p := OpenAI("test-key",
		WithOpenAIModel("gpt-4-turbo"),
		WithOpenAIMaxTokens(8192),
		WithOpenAITemperature(0.5),
		WithOpenAIBaseURL("https://azure.api"),
	)

	if p.apiKey != "test-key" {
		t.Error("API key not set")
	}
	if p.model != "gpt-4-turbo" {
		t.Error("model not set")
	}
	if p.maxTokens != 8192 {
		t.Error("maxTokens not set")
	}
	if p.temperature != 0.5 {
		t.Error("temperature not set")
	}
	if p.baseURL != "https://azure.api" {
		t.Error("baseURL not set")
	}
}

func TestOpenAIAvailable(t *testing.T) {
	p1 := OpenAI("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := OpenAI("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestOpenAIEnvKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	p := OpenAI("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

// --- DeepSeek ---

func TestDeepSeekNew(t *testing.T) {
	p := DeepSeek("")
	if p.Name() != "deepseek" {
		t.Errorf("expected 'deepseek', got %q", p.Name())
	}
	if p.model != "deepseek-chat" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.maxTokens != 4096 {
		t.Errorf("expected 4096, got %d", p.maxTokens)
	}
}

func TestDeepSeekWithOptions(t *testing.T) {
	p := DeepSeek("test-key",
		WithDeepSeekModel("deepseek-coder"),
		WithDeepSeekMaxTokens(4096),
		WithDeepSeekTemperature(0.3),
	)

	if p.apiKey != "test-key" {
		t.Error("API key not set")
	}
	if p.model != "deepseek-coder" {
		t.Error("model not set")
	}
	if p.maxTokens != 4096 {
		t.Error("maxTokens not set")
	}
	if p.temperature != 0.3 {
		t.Error("temperature not set")
	}
}

func TestDeepSeekAvailable(t *testing.T) {
	p1 := DeepSeek("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := DeepSeek("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestDeepSeekEnvKey(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "env-key")
	p := DeepSeek("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

// --- GLM ---

func TestGLMNew(t *testing.T) {
	p := GLM("")
	if p.Name() != "glm" {
		t.Errorf("expected 'glm', got %q", p.Name())
	}
	if p.model != "glm-4-flash" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.maxTokens != 4096 {
		t.Errorf("expected 4096, got %d", p.maxTokens)
	}
}

func TestGLMWithOptions(t *testing.T) {
	p := GLM("test-key",
		WithGLMModel("glm-4"),
		WithGLMMaxTokens(8192),
		WithGLMTemperature(0.5),
	)

	if p.apiKey != "test-key" {
		t.Error("API key not set")
	}
	if p.model != "glm-4" {
		t.Error("model not set")
	}
	if p.maxTokens != 8192 {
		t.Error("maxTokens not set")
	}
	if p.temperature != 0.5 {
		t.Error("temperature not set")
	}
}

func TestGLMAvailable(t *testing.T) {
	p1 := GLM("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := GLM("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestGLMEnvKey(t *testing.T) {
	t.Setenv("GLM_API_KEY", "env-key")
	p := GLM("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

// --- Local ---

func TestLocalNew(t *testing.T) {
	p := Local("")
	if p.Name() != "local" {
		t.Errorf("expected 'local', got %q", p.Name())
	}
	if p.baseURL != "http://localhost:11434/v1" {
		t.Errorf("expected default baseURL, got %q", p.baseURL)
	}
	if p.model != "llama3" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.maxTokens != 4096 {
		t.Errorf("expected 4096, got %d", p.maxTokens)
	}
}

func TestLocalWithOptions(t *testing.T) {
	p := Local("http://custom:8000/v1",
		WithLocalModel("mistral"),
		WithLocalAPIKey("secret"),
		WithLocalMaxTokens(2048),
		WithLocalTemperature(0.8),
	)

	if p.baseURL != "http://custom:8000/v1" {
		t.Error("baseURL not set")
	}
	if p.model != "mistral" {
		t.Error("model not set")
	}
	if p.apiKey != "secret" {
		t.Error("apiKey not set")
	}
	if p.maxTokens != 2048 {
		t.Error("maxTokens not set")
	}
	if p.temperature != 0.8 {
		t.Error("temperature not set")
	}
}

func TestLocalAvailable(t *testing.T) {
	p := Local("")
	if !p.Available() {
		t.Error("local should be available with default baseURL")
	}
}

func TestLocalCustomBaseURL(t *testing.T) {
	p := Local("http://my-server:9000/v1")
	if p.baseURL != "http://my-server:9000/v1" {
		t.Errorf("expected custom URL, got %q", p.baseURL)
	}
}

// --- Ollama ---

func TestOllama(t *testing.T) {
	p := Ollama("llama3.1")
	if p.baseURL != "http://localhost:11434/v1" {
		t.Error("wrong baseURL for Ollama")
	}
	if p.model != "llama3.1" {
		t.Error("model not set")
	}
	if p.Name() != "local" {
		t.Errorf("expected 'local', got %q", p.Name())
	}
}

func TestOllamaWithOptions(t *testing.T) {
	p := Ollama("codellama", WithLocalTemperature(0.1))
	if p.model != "codellama" {
		t.Error("model not set")
	}
	if p.temperature != 0.1 {
		t.Error("temperature not set")
	}
}

// --- vLLM ---

func TestVLLM(t *testing.T) {
	p := VLLM("mistral-7b")
	if p.baseURL != "http://localhost:8000/v1" {
		t.Error("wrong baseURL for vLLM")
	}
	if p.model != "mistral-7b" {
		t.Error("model not set")
	}
}

func TestVLLMWithOptions(t *testing.T) {
	p := VLLM("llama2", WithLocalMaxTokens(1024))
	if p.model != "llama2" {
		t.Error("model not set")
	}
	if p.maxTokens != 1024 {
		t.Error("maxTokens not set")
	}
}

// --- wrapOpenAIError ---

func TestWrapOpenAIErrorRateLimit(t *testing.T) {
	apiErr := &openai.Error{
		StatusCode: http.StatusTooManyRequests,
		Request:    &http.Request{},
		Response:   &http.Response{StatusCode: http.StatusTooManyRequests},
	}
	wrapped := wrapOpenAIError(apiErr)
	if !errors.Is(wrapped, allm.ErrRateLimited) {
		t.Errorf("expected ErrRateLimited, got %v", wrapped)
	}
}

func TestWrapOpenAIErrorNonRateLimit(t *testing.T) {
	apiErr := &openai.Error{
		StatusCode: http.StatusInternalServerError,
		Request:    &http.Request{},
		Response:   &http.Response{StatusCode: http.StatusInternalServerError},
	}
	wrapped := wrapOpenAIError(apiErr)
	if errors.Is(wrapped, allm.ErrRateLimited) {
		t.Error("should not be ErrRateLimited for 500")
	}
}

func TestWrapOpenAIErrorNil(t *testing.T) {
	if wrapOpenAIError(nil) != nil {
		t.Error("nil error should return nil")
	}
}

func TestWrapOpenAIErrorNonAPIError(t *testing.T) {
	err := fmt.Errorf("network error")
	wrapped := wrapOpenAIError(err)
	if errors.Is(wrapped, allm.ErrRateLimited) {
		t.Error("generic error should not be ErrRateLimited")
	}
	if wrapped.Error() != "network error" {
		t.Errorf("expected original error, got %v", wrapped)
	}
}

// --- wrapAnthropicError tests ---

func TestWrapAnthropicErrorRateLimit(t *testing.T) {
	apiErr := &anthropic.Error{
		StatusCode: http.StatusTooManyRequests,
	}
	wrapped := wrapAnthropicError(apiErr)
	if !errors.Is(wrapped, allm.ErrRateLimited) {
		t.Errorf("expected ErrRateLimited, got %v", wrapped)
	}
}

func TestWrapAnthropicErrorNonRateLimit(t *testing.T) {
	apiErr := &anthropic.Error{
		StatusCode: http.StatusInternalServerError,
	}
	wrapped := wrapAnthropicError(apiErr)
	if errors.Is(wrapped, allm.ErrRateLimited) {
		t.Error("should not be ErrRateLimited for 500")
	}
}

func TestWrapAnthropicErrorNil(t *testing.T) {
	if wrapAnthropicError(nil) != nil {
		t.Error("nil error should return nil")
	}
}

func TestWrapAnthropicErrorNonAPIError(t *testing.T) {
	err := fmt.Errorf("network error")
	wrapped := wrapAnthropicError(err)
	if errors.Is(wrapped, allm.ErrRateLimited) {
		t.Error("generic error should not be ErrRateLimited")
	}
	if wrapped.Error() != "network error" {
		t.Errorf("expected original error, got %v", wrapped)
	}
}

// --- convertToOpenAI tests ---

func TestConvertToOpenAI(t *testing.T) {
	msgs := []allm.Message{
		{Role: allm.RoleSystem, Content: "You are helpful."},
		{Role: allm.RoleUser, Content: "Hello"},
		{Role: allm.RoleAssistant, Content: "Hi there!"},
	}

	result := convertToOpenAI(msgs)
	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result))
	}
	// System message
	if result[0].OfSystem == nil {
		t.Error("expected system message")
	}
	// User message
	if result[1].OfUser == nil {
		t.Error("expected user message")
	}
	// Assistant message
	if result[2].OfAssistant == nil {
		t.Error("expected assistant message")
	}
}

func TestConvertToOpenAIToolCalls(t *testing.T) {
	msgs := []allm.Message{
		{
			Role: allm.RoleAssistant,
			ToolCalls: []allm.ToolCall{
				{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Tokyo"}`)},
			},
		},
	}

	result := convertToOpenAI(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfAssistant == nil {
		t.Fatal("expected assistant message")
	}
	if len(result[0].OfAssistant.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result[0].OfAssistant.ToolCalls))
	}
	if result[0].OfAssistant.ToolCalls[0].OfFunction.Function.Name != "get_weather" {
		t.Error("expected get_weather function name")
	}
}

func TestConvertToOpenAIToolResults(t *testing.T) {
	msgs := []allm.Message{
		{
			Role: allm.RoleTool,
			ToolResults: []allm.ToolResult{
				{ToolCallID: "call_1", Content: "32°C sunny"},
				{ToolCallID: "call_2", Content: "rainy"},
			},
		},
	}

	result := convertToOpenAI(msgs)
	// Each tool result becomes a separate tool message
	if len(result) != 2 {
		t.Fatalf("expected 2 tool messages, got %d", len(result))
	}
	if result[0].OfTool == nil {
		t.Error("expected tool message")
	}
}

// --- convertToolsToOpenAI tests ---

func TestConvertToolsToOpenAI(t *testing.T) {
	tools := []allm.Tool{
		{
			Name:        "get_weather",
			Description: "Get weather for a city",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		},
		{
			Name:        "search",
			Description: "Search the web",
			Parameters:  map[string]any{"type": "object"},
		},
	}

	result := convertToolsToOpenAI(tools)
	if len(result) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(result))
	}
	if result[0].OfFunction.Function.Name != "get_weather" {
		t.Errorf("expected get_weather, got %q", result[0].OfFunction.Function.Name)
	}
	if result[1].OfFunction.Function.Name != "search" {
		t.Errorf("expected search, got %q", result[1].OfFunction.Function.Name)
	}
}

// --- toStringSlice tests ---

func TestToStringSliceNil(t *testing.T) {
	result := toStringSlice(nil)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestToStringSliceValid(t *testing.T) {
	input := []any{"city", "country", "zip"}
	result := toStringSlice(input)
	if len(result) != 3 {
		t.Fatalf("expected 3 items, got %d", len(result))
	}
	if result[0] != "city" || result[1] != "country" || result[2] != "zip" {
		t.Errorf("unexpected result: %v", result)
	}
}

func TestToStringSliceMixedTypes(t *testing.T) {
	input := []any{"city", 42, "zip", true}
	result := toStringSlice(input)
	// Only strings should be included
	if len(result) != 2 {
		t.Fatalf("expected 2 string items, got %d", len(result))
	}
	if result[0] != "city" || result[1] != "zip" {
		t.Errorf("unexpected result: %v", result)
	}
}

func TestToStringSliceNotArray(t *testing.T) {
	result := toStringSlice("not an array")
	if result != nil {
		t.Errorf("expected nil for non-array input, got %v", result)
	}
}

// --- openaiCompleteResponse tests ---

func TestOpenAICompleteResponse(t *testing.T) {
	completion := &openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Content: "Hello!",
					ToolCalls: []openai.ChatCompletionMessageToolCallUnion{
						{
							ID:   "call_1",
							Type: "function",
							Function: openai.ChatCompletionMessageFunctionToolCallFunction{
								Name:      "fn1",
								Arguments: `{"key":"val"}`,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
		},
	}

	start := time.Now()
	resp, err := openaiCompleteResponse(completion, "test", "test-model", start)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Content)
	}
	if resp.Provider != "test" {
		t.Errorf("expected provider 'test', got %q", resp.Provider)
	}
	if resp.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", resp.Model)
	}
	if resp.InputTokens != 10 {
		t.Errorf("expected 10 input tokens, got %d", resp.InputTokens)
	}
	if resp.OutputTokens != 5 {
		t.Errorf("expected 5 output tokens, got %d", resp.OutputTokens)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "fn1" {
		t.Errorf("expected fn1, got %q", resp.ToolCalls[0].Name)
	}
}

func TestOpenAICompleteResponseEmpty(t *testing.T) {
	completion := &openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{},
	}

	_, err := openaiCompleteResponse(completion, "test", "test-model", time.Now())
	if !errors.Is(err, allm.ErrEmptyResponse) {
		t.Errorf("expected ErrEmptyResponse, got %v", err)
	}
}
