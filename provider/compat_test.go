package provider

import (
	"testing"

	"github.com/kusandriadi/allm-go"
)

// --- DeepSeek (new implementation) ---

func TestDeepSeekCompat(t *testing.T) {
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
	if p.baseURL != "https://api.deepseek.com/v1" {
		t.Errorf("expected DeepSeek base URL, got %q", p.baseURL)
	}
}

func TestDeepSeekCompatWithOptions(t *testing.T) {
	p := DeepSeek("test-key",
		WithDefaultModel("deepseek-coder"),
		WithMaxTokens(8192),
		WithTemperature(0.3),
	)

	if p.apiKey != "test-key" {
		t.Error("API key not set")
	}
	if p.model != "deepseek-coder" {
		t.Error("model not set")
	}
	if p.maxTokens != 8192 {
		t.Error("maxTokens not set")
	}
	if p.temperature != 0.3 {
		t.Error("temperature not set")
	}
}

func TestDeepSeekCompatAvailable(t *testing.T) {
	p1 := DeepSeek("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := DeepSeek("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestDeepSeekCompatEnvKey(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "env-key")
	p := DeepSeek("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

// --- GLM (new implementation) ---

func TestGLMCompat(t *testing.T) {
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
	if p.baseURL != "https://open.bigmodel.cn/api/paas/v4/" {
		t.Errorf("expected GLM base URL, got %q", p.baseURL)
	}
	if p.embedModel != "embedding-3" {
		t.Errorf("expected default embedding model, got %q", p.embedModel)
	}
}

func TestGLMCompatWithOptions(t *testing.T) {
	p := GLM("test-key",
		WithDefaultModel("glm-4"),
		WithMaxTokens(8192),
		WithTemperature(0.5),
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

func TestGLMCompatAvailable(t *testing.T) {
	p1 := GLM("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := GLM("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestGLMCompatEnvKey(t *testing.T) {
	t.Setenv("GLM_API_KEY", "env-key")
	p := GLM("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

// --- Gemini ---

func TestGemini(t *testing.T) {
	p := Gemini("")
	if p.Name() != "gemini" {
		t.Errorf("expected 'gemini', got %q", p.Name())
	}
	if p.model != "gemini-2.0-flash" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.baseURL != "https://generativelanguage.googleapis.com/v1beta/openai/" {
		t.Errorf("expected Gemini base URL, got %q", p.baseURL)
	}
}

func TestGeminiWithOptions(t *testing.T) {
	p := Gemini("test-key",
		WithDefaultModel("gemini-1.5-pro"),
		WithMaxTokens(2048),
		WithTemperature(0.9),
	)

	if p.apiKey != "test-key" {
		t.Error("API key not set")
	}
	if p.model != "gemini-1.5-pro" {
		t.Error("model not set")
	}
	if p.maxTokens != 2048 {
		t.Error("maxTokens not set")
	}
	if p.temperature != 0.9 {
		t.Error("temperature not set")
	}
}

func TestGeminiAvailable(t *testing.T) {
	p1 := Gemini("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := Gemini("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestGeminiEnvKey(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "env-key")
	p := Gemini("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

// --- Groq ---

func TestGroq(t *testing.T) {
	p := Groq("")
	if p.Name() != "groq" {
		t.Errorf("expected 'groq', got %q", p.Name())
	}
	if p.model != "llama-3.3-70b-versatile" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.baseURL != "https://api.groq.com/openai/v1" {
		t.Errorf("expected Groq base URL, got %q", p.baseURL)
	}
}

func TestGroqWithOptions(t *testing.T) {
	p := Groq("test-key",
		WithDefaultModel("mixtral-8x7b-32768"),
		WithTemperature(0.2),
	)

	if p.apiKey != "test-key" {
		t.Error("API key not set")
	}
	if p.model != "mixtral-8x7b-32768" {
		t.Error("model not set")
	}
	if p.temperature != 0.2 {
		t.Error("temperature not set")
	}
}

func TestGroqAvailable(t *testing.T) {
	p1 := Groq("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := Groq("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestGroqEnvKey(t *testing.T) {
	t.Setenv("GROQ_API_KEY", "env-key")
	p := Groq("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

// --- Perplexity ---

func TestPerplexity(t *testing.T) {
	p := Perplexity("")
	if p.Name() != "perplexity" {
		t.Errorf("expected 'perplexity', got %q", p.Name())
	}
	if p.model != "llama-3.1-sonar-small-128k-online" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.baseURL != "https://api.perplexity.ai" {
		t.Errorf("expected Perplexity base URL, got %q", p.baseURL)
	}
}

func TestPerplexityWithOptions(t *testing.T) {
	p := Perplexity("test-key",
		WithDefaultModel("llama-3.1-sonar-large-128k-online"),
		WithMaxTokens(1024),
	)

	if p.apiKey != "test-key" {
		t.Error("API key not set")
	}
	if p.model != "llama-3.1-sonar-large-128k-online" {
		t.Error("model not set")
	}
	if p.maxTokens != 1024 {
		t.Error("maxTokens not set")
	}
}

func TestPerplexityAvailable(t *testing.T) {
	p1 := Perplexity("")
	if p1.Available() {
		t.Error("should not be available without key")
	}

	p2 := Perplexity("test-key")
	if !p2.Available() {
		t.Error("should be available with key")
	}
}

func TestPerplexityEnvKey(t *testing.T) {
	t.Setenv("PERPLEXITY_API_KEY", "env-key")
	p := Perplexity("")
	if p.apiKey != "env-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

// --- Local (new implementation) ---

func TestLocalCompat(t *testing.T) {
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

func TestLocalCompatWithOptions(t *testing.T) {
	p := Local("http://custom:8000/v1",
		WithDefaultModel("mistral"),
		WithMaxTokens(2048),
		WithTemperature(0.8),
	)

	if p.baseURL != "http://custom:8000/v1" {
		t.Error("baseURL not set")
	}
	if p.model != "mistral" {
		t.Error("model not set")
	}
	if p.maxTokens != 2048 {
		t.Error("maxTokens not set")
	}
	if p.temperature != 0.8 {
		t.Error("temperature not set")
	}
}

func TestLocalCompatAvailable(t *testing.T) {
	p := Local("")
	if !p.Available() {
		t.Error("local should be available with default baseURL")
	}
}

// --- Ollama (new implementation) ---

func TestOllamaCompat(t *testing.T) {
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

func TestOllamaCompatWithOptions(t *testing.T) {
	p := Ollama("codellama", WithTemperature(0.1))
	if p.model != "codellama" {
		t.Error("model not set")
	}
	if p.temperature != 0.1 {
		t.Error("temperature not set")
	}
}

// --- vLLM (new implementation) ---

func TestVLLMCompat(t *testing.T) {
	p := VLLM("mistral-7b")
	if p.baseURL != "http://localhost:8000/v1" {
		t.Error("wrong baseURL for vLLM")
	}
	if p.model != "mistral-7b" {
		t.Error("model not set")
	}
}

func TestVLLMCompatWithOptions(t *testing.T) {
	p := VLLM("llama2", WithMaxTokens(1024))
	if p.model != "llama2" {
		t.Error("model not set")
	}
	if p.maxTokens != 1024 {
		t.Error("maxTokens not set")
	}
}

// --- OpenAICompatible generic constructor ---

func TestOpenAICompatibleCustomProvider(t *testing.T) {
	// Create a custom provider not in the registry
	p := OpenAICompatible("custom", "my-key",
		WithBaseURL("https://custom.api/v1"),
		WithDefaultModel("custom-model"),
		WithMaxTokens(2048),
	)

	if p.Name() != "custom" {
		t.Errorf("expected 'custom', got %q", p.Name())
	}
	if p.apiKey != "my-key" {
		t.Error("API key not set")
	}
	if p.baseURL != "https://custom.api/v1" {
		t.Error("baseURL not set")
	}
	if p.model != "custom-model" {
		t.Error("model not set")
	}
	if p.maxTokens != 2048 {
		t.Error("maxTokens not set")
	}
}

func TestOpenAICompatibleRegistryLookup(t *testing.T) {
	// Test that registry defaults are applied
	p := OpenAICompatible(allm.DeepSeek, "test-key")

	if p.baseURL != "https://api.deepseek.com/v1" {
		t.Errorf("expected DeepSeek base URL from registry, got %q", p.baseURL)
	}
	if p.model != "deepseek-chat" {
		t.Errorf("expected default model from registry, got %q", p.model)
	}
}

func TestOpenAICompatibleExplicitOverridesRegistry(t *testing.T) {
	// Test that explicit options override registry defaults
	p := OpenAICompatible(allm.DeepSeek, "test-key",
		WithBaseURL("https://custom.api"),
		WithDefaultModel("custom-model"),
	)

	if p.baseURL != "https://custom.api" {
		t.Error("explicit baseURL should override registry")
	}
	if p.model != "custom-model" {
		t.Error("explicit model should override registry")
	}
}

func TestOpenAICompatibleEnvKeyFallback(t *testing.T) {
	t.Setenv("GROQ_API_KEY", "env-groq-key")
	p := OpenAICompatible(allm.Groq, "")

	if p.apiKey != "env-groq-key" {
		t.Errorf("expected env key, got %q", p.apiKey)
	}
}

func TestOpenAICompatibleExplicitKeyOverridesEnv(t *testing.T) {
	t.Setenv("GROQ_API_KEY", "env-groq-key")
	p := OpenAICompatible(allm.Groq, "explicit-key")

	if p.apiKey != "explicit-key" {
		t.Errorf("expected explicit key to override env, got %q", p.apiKey)
	}
}

// --- Embed support tests ---

func TestEmbedSupportGLM(t *testing.T) {
	p := GLM("test-key")
	if p.embedModel != "embedding-3" {
		t.Errorf("expected default embedding model for GLM, got %q", p.embedModel)
	}
}

func TestEmbedSupportLocal(t *testing.T) {
	p := Local("llama3")
	if p.embedModel != "llama3" {
		t.Errorf("expected embedding model to match chat model for Local, got %q", p.embedModel)
	}
}

func TestEmbedSupportCustomModel(t *testing.T) {
	p := GLM("test-key", WithEmbedModel("custom-embed"))
	if p.embedModel != "custom-embed" {
		t.Errorf("expected custom embedding model, got %q", p.embedModel)
	}
}
