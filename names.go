package allm

// ProviderName represents the name of an LLM provider.
type ProviderName string

// Provider name constants.
const (
	// Anthropic is the name for Anthropic Claude models.
	Anthropic ProviderName = "anthropic"
	// OpenAI is the name for OpenAI GPT models.
	OpenAI ProviderName = "openai"
	// DeepSeek is the name for DeepSeek models.
	DeepSeek ProviderName = "deepseek"
	// Gemini is the name for Google Gemini models.
	Gemini ProviderName = "gemini"
	// Groq is the name for Groq models.
	Groq ProviderName = "groq"
	// GLM is the name for Zhipu AI GLM models.
	GLM ProviderName = "glm"
	// Perplexity is the name for Perplexity models.
	Perplexity ProviderName = "perplexity"
	// Local is the name for local/self-hosted models (Ollama, vLLM, etc).
	Local ProviderName = "local"
)
