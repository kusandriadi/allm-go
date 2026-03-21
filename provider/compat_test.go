package provider

import (
	"testing"

	"github.com/kusandriadi/allm-go"
)

// providerTestCase defines expected defaults for a registered provider shortcut.
type providerTestCase struct {
	name       string
	create     func(string, ...CompatOption) *OpenAICompatibleProvider
	wantName   string
	wantModel  string
	wantURL    string
	envKey     string
	embedModel string // non-empty if provider supports embeddings
}

// registeredProviders lists all OpenAI-compatible provider shortcuts with expected defaults.
var registeredProviders = []providerTestCase{
	{
		name: "DeepSeek", create: DeepSeek,
		wantName: "deepseek", wantModel: "deepseek-chat",
		wantURL: "https://api.deepseek.com/v1", envKey: "DEEPSEEK_API_KEY",
	},
	{
		name: "GLM", create: GLM,
		wantName: "glm", wantModel: "glm-4-flash",
		wantURL: "https://open.bigmodel.cn/api/paas/v4/", envKey: "GLM_API_KEY",
		embedModel: "embedding-3",
	},
	{
		name: "Gemini", create: Gemini,
		wantName: "gemini", wantModel: "gemini-2.0-flash",
		wantURL: "https://generativelanguage.googleapis.com/v1beta/openai/", envKey: "GEMINI_API_KEY",
	},
	{
		name: "Kimi", create: Kimi,
		wantName: "kimi", wantModel: "moonshot-v1-8k",
		wantURL: "https://api.moonshot.cn/v1", envKey: "MOONSHOT_API_KEY",
	},
	{
		name: "Qwen", create: Qwen,
		wantName: "qwen", wantModel: "qwen-plus",
		wantURL: "https://dashscope.aliyuncs.com/compatible-mode/v1", envKey: "DASHSCOPE_API_KEY",
		embedModel: "text-embedding-v3",
	},
	{
		name: "MiniMax", create: MiniMax,
		wantName: "minimax", wantModel: "MiniMax-Text-01",
		wantURL: "https://api.minimax.chat/v1", envKey: "MINIMAX_API_KEY",
	},
}

// TestProviderDefaults verifies each shortcut returns correct name, model, URL, and maxTokens.
func TestProviderDefaults(t *testing.T) {
	for _, tc := range registeredProviders {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.create("")
			if p.Name() != tc.wantName {
				t.Errorf("Name() = %q, want %q", p.Name(), tc.wantName)
			}
			if p.model != tc.wantModel {
				t.Errorf("model = %q, want %q", p.model, tc.wantModel)
			}
			if p.baseURL != tc.wantURL {
				t.Errorf("baseURL = %q, want %q", p.baseURL, tc.wantURL)
			}
			if p.maxTokens != 4096 {
				t.Errorf("maxTokens = %d, want 4096", p.maxTokens)
			}
			if tc.embedModel != "" && p.embedModel != tc.embedModel {
				t.Errorf("embedModel = %q, want %q", p.embedModel, tc.embedModel)
			}
		})
	}
}

// TestProviderWithOptions verifies custom options override defaults.
func TestProviderWithOptions(t *testing.T) {
	for _, tc := range registeredProviders {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.create("test-key",
				WithDefaultModel("custom-model"),
				WithMaxTokens(8192),
				WithTemperature(0.5),
			)
			if p.apiKey != "test-key" {
				t.Error("apiKey not set")
			}
			if p.model != "custom-model" {
				t.Error("model override not applied")
			}
			if p.maxTokens != 8192 {
				t.Error("maxTokens override not applied")
			}
			if p.temperature != 0.5 {
				t.Error("temperature override not applied")
			}
		})
	}
}

// TestProviderAvailable verifies availability checks (key required for non-local).
func TestProviderAvailable(t *testing.T) {
	for _, tc := range registeredProviders {
		t.Run(tc.name+"_without_key", func(t *testing.T) {
			if tc.create("").Available() {
				t.Error("should not be available without key")
			}
		})
		t.Run(tc.name+"_with_key", func(t *testing.T) {
			if !tc.create("test-key").Available() {
				t.Error("should be available with key")
			}
		})
	}
}

// TestProviderEnvKey verifies each shortcut reads from the correct environment variable.
func TestProviderEnvKey(t *testing.T) {
	for _, tc := range registeredProviders {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(tc.envKey, "env-key")
			p := tc.create("")
			if p.apiKey != "env-key" {
				t.Errorf("apiKey = %q, want env-key (from %s)", p.apiKey, tc.envKey)
			}
		})
	}
}

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
	p := OpenAICompatible(allm.DeepSeek, "test-key")
	if p.baseURL != "https://api.deepseek.com/v1" {
		t.Errorf("expected registry baseURL, got %q", p.baseURL)
	}
	if p.model != "deepseek-chat" {
		t.Errorf("expected registry model, got %q", p.model)
	}
}

func TestOpenAICompatibleExplicitOverridesRegistry(t *testing.T) {
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
	t.Setenv("DEEPSEEK_API_KEY", "env-ds-key")
	p := OpenAICompatible(allm.DeepSeek, "")
	if p.apiKey != "env-ds-key" {
		t.Errorf("apiKey = %q, want env-ds-key", p.apiKey)
	}
}

func TestOpenAICompatibleExplicitKeyOverridesEnv(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "env-ds-key")
	p := OpenAICompatible(allm.DeepSeek, "explicit-key")
	if p.apiKey != "explicit-key" {
		t.Errorf("apiKey = %q, want explicit-key", p.apiKey)
	}
}

// --- Embed support ---

func TestEmbedSupportGLM(t *testing.T) {
	p := GLM("test-key")
	if p.embedModel != "embedding-3" {
		t.Errorf("embedModel = %q, want embedding-3", p.embedModel)
	}
}

func TestEmbedSupportLocal(t *testing.T) {
	p := Ollama("llama3")
	if p.embedModel != "llama3" {
		t.Errorf("embedModel = %q, want llama3 (should match chat model)", p.embedModel)
	}
}

func TestEmbedSupportCustomModel(t *testing.T) {
	p := GLM("test-key", WithEmbedModel("custom-embed"))
	if p.embedModel != "custom-embed" {
		t.Errorf("embedModel = %q, want custom-embed", p.embedModel)
	}
}
