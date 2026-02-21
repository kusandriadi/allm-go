package provider

import "github.com/kusandriadi/allm-go"

// DeepSeek creates a new DeepSeek provider.
// If apiKey is empty, it reads from DEEPSEEK_API_KEY environment variable.
func DeepSeek(apiKey string, opts ...CompatOption) *OpenAICompatibleProvider {
	return OpenAICompatible(allm.DeepSeek, apiKey, opts...)
}

// Gemini creates a new Google Gemini provider.
// If apiKey is empty, it reads from GEMINI_API_KEY environment variable.
func Gemini(apiKey string, opts ...CompatOption) *OpenAICompatibleProvider {
	return OpenAICompatible(allm.Gemini, apiKey, opts...)
}

// Groq creates a new Groq provider.
// If apiKey is empty, it reads from GROQ_API_KEY environment variable.
func Groq(apiKey string, opts ...CompatOption) *OpenAICompatibleProvider {
	return OpenAICompatible(allm.Groq, apiKey, opts...)
}

// GLM creates a new Zhipu AI GLM provider.
// If apiKey is empty, it reads from GLM_API_KEY environment variable.
func GLM(apiKey string, opts ...CompatOption) *OpenAICompatibleProvider {
	return OpenAICompatible(allm.GLM, apiKey, opts...)
}

// Perplexity creates a new Perplexity provider.
// If apiKey is empty, it reads from PERPLEXITY_API_KEY environment variable.
func Perplexity(apiKey string, opts ...CompatOption) *OpenAICompatibleProvider {
	return OpenAICompatible(allm.Perplexity, apiKey, opts...)
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
