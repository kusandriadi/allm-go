package provider

import (
	"os"

	"github.com/kusandriadi/allm-go"
)

// GLM creates a new Zhipu AI GLM provider using the Anthropic-compatible API.
// If apiKey is empty, it reads from GLM_API_KEY environment variable.
func GLM(apiKey string, opts ...AnthropicOption) *AnthropicProvider {
	if apiKey == "" {
		apiKey = os.Getenv("GLM_API_KEY")
	}
	defaults := []AnthropicOption{
		WithAnthropicBaseURL("https://api.z.ai/api/anthropic"),
		WithAnthropicModel(GLM4Dot7),
	}
	allOpts := append(defaults, opts...)
	p := Anthropic(apiKey, allOpts...)
	p.name = "glm"
	return p
}

// Kimi creates a new Moonshot AI Kimi provider using the Anthropic-compatible API.
// If apiKey is empty, it reads from MOONSHOT_API_KEY environment variable.
func Kimi(apiKey string, opts ...AnthropicOption) *AnthropicProvider {
	if apiKey == "" {
		apiKey = os.Getenv("MOONSHOT_API_KEY")
	}
	defaults := []AnthropicOption{
		WithAnthropicBaseURL("https://api.moonshot.ai/anthropic"),
		WithAnthropicModel(KimiK2_5),
	}
	allOpts := append(defaults, opts...)
	p := Anthropic(apiKey, allOpts...)
	p.name = "kimi"
	return p
}

// MiniMax creates a new MiniMax provider using the Anthropic-compatible API.
// If apiKey is empty, it reads from MINIMAX_API_KEY environment variable.
func MiniMax(apiKey string, opts ...AnthropicOption) *AnthropicProvider {
	if apiKey == "" {
		apiKey = os.Getenv("MINIMAX_API_KEY")
	}
	defaults := []AnthropicOption{
		WithAnthropicBaseURL("https://api.minimax.io/anthropic"),
		WithAnthropicModel(MiniMaxM2_7),
	}
	allOpts := append(defaults, opts...)
	p := Anthropic(apiKey, allOpts...)
	p.name = "minimax"
	return p
}

// Local creates a Local provider for OpenAI-compatible servers.
// The baseURL parameter specifies the server endpoint (e.g., "http://localhost:11434/v1").
// If baseURL is empty, defaults to http://localhost:11434/v1 (Ollama).
// Use WithDefaultModel to set the model name.
func Local(baseURL string, opts ...CompatOption) *OpenAICompatibleProvider {
	if baseURL != "" {
		opts = append([]CompatOption{WithBaseURL(baseURL)}, opts...)
	}
	return OpenAICompatible(allm.Local, "", opts...)
}

// Ollama creates a Local provider configured for Ollama.
// The model parameter sets the default model to use (e.g., "llama3", "mistral").
func Ollama(model string, opts ...CompatOption) *OpenAICompatibleProvider {
	opts = append([]CompatOption{
		WithDefaultModel(model),
		WithBaseURL("http://localhost:11434/v1"),
	}, opts...)
	return OpenAICompatible(allm.Local, "", opts...)
}

// VLLM creates a Local provider configured for vLLM.
// The model parameter sets the default model to use.
func VLLM(model string, opts ...CompatOption) *OpenAICompatibleProvider {
	opts = append([]CompatOption{
		WithDefaultModel(model),
		WithBaseURL("http://localhost:8000/v1"),
	}, opts...)
	return OpenAICompatible(allm.Local, "", opts...)
}
