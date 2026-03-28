package provider

import (
	"testing"

	"github.com/kusandriadi/allm-go"
)

// --- Local / Ollama / vLLM ---

func TestLocal(t *testing.T) {
	p := Local("")
	if p.Name() != "local" {
		t.Errorf("Name() = %q, want local", p.Name())
	}
	if p.baseURL != "http://localhost:11434/v1" {
		t.Errorf("baseURL = %q, want default Ollama URL", p.baseURL)
	}
	if p.model != "llama3" {
		t.Errorf("model = %q, want llama3", p.model)
	}
	if !p.Available() {
		t.Error("local should be available with default baseURL")
	}
}

func TestLocalWithCustomURL(t *testing.T) {
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
}

func TestOllama(t *testing.T) {
	p := Ollama("llama3.1")
	if p.Name() != "local" {
		t.Errorf("Name() = %q, want local", p.Name())
	}
	if p.baseURL != "http://localhost:11434/v1" {
		t.Errorf("baseURL = %q, want Ollama URL", p.baseURL)
	}
	if p.model != "llama3.1" {
		t.Errorf("model = %q, want llama3.1", p.model)
	}
}

func TestOllamaWithOptions(t *testing.T) {
	p := Ollama("codellama", WithTemperature(0.1))
	if p.model != "codellama" {
		t.Error("model not set")
	}
	if p.temperature != 0.1 {
		t.Error("temperature not set")
	}
}

func TestVLLM(t *testing.T) {
	p := VLLM("mistral-7b")
	if p.baseURL != "http://localhost:8000/v1" {
		t.Errorf("baseURL = %q, want vLLM URL", p.baseURL)
	}
	if p.model != "mistral-7b" {
		t.Errorf("model = %q, want mistral-7b", p.model)
	}
}

func TestVLLMWithOptions(t *testing.T) {
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
	p := OpenAICompatible("custom", "my-key",
		WithBaseURL("https://custom.api/v1"),
		WithDefaultModel("custom-model"),
		WithMaxTokens(2048),
	)
	if p.Name() != "custom" {
		t.Errorf("Name() = %q, want custom", p.Name())
	}
	if p.apiKey != "my-key" {
		t.Error("apiKey not set")
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
	p := OpenAICompatible(allm.Local, "test-key")
	if p.baseURL != "http://localhost:11434/v1" {
		t.Errorf("expected registry baseURL, got %q", p.baseURL)
	}
	if p.model != "llama3" {
		t.Errorf("expected registry model, got %q", p.model)
	}
}

func TestOpenAICompatibleExplicitOverridesRegistry(t *testing.T) {
	p := OpenAICompatible(allm.Local, "test-key",
		WithBaseURL("http://localhost:9000/v1"),
		WithDefaultModel("custom-model"),
	)
	if p.baseURL != "http://localhost:9000/v1" {
		t.Error("explicit baseURL should override registry")
	}
	if p.model != "custom-model" {
		t.Error("explicit model should override registry")
	}
}

func TestOpenAICompatibleEnvKeyFallback(t *testing.T) {
	t.Setenv("LOCAL_API_KEY", "env-local-key")
	p := OpenAICompatible(allm.Local, "")
	if p.apiKey != "env-local-key" {
		t.Errorf("apiKey = %q, want env-local-key", p.apiKey)
	}
}

func TestOpenAICompatibleExplicitKeyOverridesEnv(t *testing.T) {
	t.Setenv("LOCAL_API_KEY", "env-local-key")
	p := OpenAICompatible(allm.Local, "explicit-key")
	if p.apiKey != "explicit-key" {
		t.Errorf("apiKey = %q, want explicit-key", p.apiKey)
	}
}

// --- Embed support ---

func TestGLMAnthropicProvider(t *testing.T) {
	p := GLM("test-key")
	if p.Name() != "glm" {
		t.Errorf("Name() = %q, want glm", p.Name())
	}
	if p.model != GLM4Dot7 {
		t.Errorf("model = %q, want %q", p.model, GLM4Dot7)
	}
	if p.baseURL != "https://api.z.ai/api/anthropic" {
		t.Errorf("baseURL = %q, want https://api.z.ai/api/anthropic", p.baseURL)
	}
}

func TestEmbedSupportLocal(t *testing.T) {
	p := Ollama("llama3")
	if p.embedModel != "llama3" {
		t.Errorf("embedModel = %q, want llama3 (should match chat model)", p.embedModel)
	}
}

func TestKimiAnthropicProvider(t *testing.T) {
	p := Kimi("test-key")
	if p.Name() != "kimi" {
		t.Errorf("Name() = %q, want kimi", p.Name())
	}
	if p.model != KimiK2_5 {
		t.Errorf("model = %q, want %q", p.model, KimiK2_5)
	}
	if p.baseURL != "https://api.moonshot.ai/anthropic" {
		t.Errorf("baseURL = %q, want https://api.moonshot.ai/anthropic", p.baseURL)
	}
}

func TestMiniMaxAnthropicProvider(t *testing.T) {
	p := MiniMax("test-key")
	if p.Name() != "minimax" {
		t.Errorf("Name() = %q, want minimax", p.Name())
	}
	if p.model != MiniMaxM2_7 {
		t.Errorf("model = %q, want %q", p.model, MiniMaxM2_7)
	}
	if p.baseURL != "https://api.minimax.io/anthropic" {
		t.Errorf("baseURL = %q, want https://api.minimax.io/anthropic", p.baseURL)
	}
}

func TestGLMCustomModel(t *testing.T) {
	p := GLM("test-key", WithAnthropicModel(GLM5))
	if p.model != "glm-5" {
		t.Errorf("model = %q, want glm-5", p.model)
	}
}
