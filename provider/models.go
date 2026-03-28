package provider

// Anthropic Claude models — always points to the latest version.
const (
	AnthropicOpus     = "opus"
	AnthropicSonnet   = "sonnet"
	AnthropicHaiku    = "haiku"
	AnthropicOpusPlan = "opusplan" // Opus for planning, Sonnet for execution
)

// OpenAI GPT models.
const (
	// GPT-5.2 - Latest and most capable
	OpenAIGPT5_2 = "gpt-5.2"
	// GPT-5.2 Pro
	OpenAIGPT5_2Pro = "gpt-5.2-pro"
	// GPT-5.1
	OpenAIGPT5_1 = "gpt-5.1"
	// GPT-5.1 Mini
	OpenAIGPT5_1Mini = "gpt-5.1-mini"
	// GPT-5
	OpenAIGPT5 = "gpt-5"
	// GPT-5 Mini
	OpenAIGPT5Mini = "gpt-5-mini"
	// GPT-5 Nano
	OpenAIGPT5Nano = "gpt-5-nano"
	// GPT-4.1
	OpenAIGPT4_1 = "gpt-4.1"
	// GPT-4.1 Mini
	OpenAIGPT4_1Mini = "gpt-4.1-mini"
	// GPT-4.1 Nano
	OpenAIGPT4_1Nano = "gpt-4.1-nano"
	// GPT-4o
	OpenAIGPT4o = "gpt-4o"
	// GPT-4o Mini
	OpenAIGPT4oMini = "gpt-4o-mini"
	// GPT-4 Turbo
	OpenAIGPT4Turbo = "gpt-4-turbo"
	// o4 Mini - Reasoning model
	OpenAIO4Mini = "o4-mini"
	// o3 - Reasoning model
	OpenAIO3 = "o3"
	// o3 Mini - Reasoning model
	OpenAIO3Mini = "o3-mini"
	// o1 - Reasoning model
	OpenAIO1 = "o1"
)

// GLM (Zhipu AI) models.
const (
	// GLM-5.1 - Latest and most capable
	GLM5Dot1 = "glm-5.1"
	// GLM-5
	GLM5 = "glm-5"
	// GLM-5 Turbo - Fast variant of GLM-5
	GLM5Turbo = "glm-5-turbo"
	// GLM-4.7
	GLM4Dot7 = "glm-4.7"
	// GLM-4.6
	GLM4Dot6 = "glm-4.6"
)

// OpenAI Embedding models.
const (
	// Text Embedding 3 Small - Fast and cost-effective
	OpenAITextEmbedding3Small = "text-embedding-3-small"
	// Text Embedding 3 Large - Higher quality
	OpenAITextEmbedding3Large = "text-embedding-3-large"
	// Text Embedding Ada 002 (legacy)
	OpenAITextEmbeddingAda002 = "text-embedding-ada-002"
)

// Kimi (Moonshot AI) models.
const (
	KimiK2_5            = "kimi-k2.5"
	KimiK2Preview       = "kimi-k2-0905-preview"
	KimiK2TurboPreview  = "kimi-k2-turbo-preview"
	KimiK2Thinking      = "kimi-k2-thinking"
	KimiK2ThinkingTurbo = "kimi-k2-thinking-turbo"
)

// MiniMax models.
const (
	MiniMaxM2_7          = "MiniMax-M2.7"
	MiniMaxM2_7HighSpeed = "MiniMax-M2.7-highspeed"
	MiniMaxM2_5          = "MiniMax-M2.5"
	MiniMaxM2_5HighSpeed = "MiniMax-M2.5-highspeed"
	MiniMaxM2            = "MiniMax-M2"
)
