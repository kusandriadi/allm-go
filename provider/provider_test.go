package provider

import (
	"errors"
	"fmt"
	"net/http"
	"testing"

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
