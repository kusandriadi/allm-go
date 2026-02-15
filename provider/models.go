package provider

// Anthropic Claude models.
// Use these constants instead of hardcoding model ID strings.
// For the latest models at runtime, use client.Models(ctx).
const (
	// Claude Opus 4.6 - Most capable model
	AnthropicClaudeOpus4_6 = "claude-opus-4-6"
	// Claude Opus 4.5
	AnthropicClaudeOpus4_5 = "claude-opus-4-5"
	// Claude Opus 4.1
	AnthropicClaudeOpus4_1 = "claude-opus-4-1-20250805"
	// Claude Opus 4
	AnthropicClaudeOpus4 = "claude-opus-4-20250514"
	// Claude Sonnet 4.5
	AnthropicClaudeSonnet4_5 = "claude-sonnet-4-5-20250929"
	// Claude Sonnet 4
	AnthropicClaudeSonnet4 = "claude-sonnet-4-20250514"
	// Claude Sonnet 3.7
	AnthropicClaudeSonnet3_7 = "claude-3-7-sonnet-20250219"
	// Claude Haiku 4.5
	AnthropicClaudeHaiku4_5 = "claude-haiku-4-5-20251001"
	// Claude Haiku 3.5
	AnthropicClaudeHaiku3_5 = "claude-3-5-haiku-20241022"
	// Claude 3 Haiku (legacy)
	AnthropicClaudeHaiku3 = "claude-3-haiku-20240307"
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

// DeepSeek models.
const (
	// DeepSeek Chat (V3) - General purpose
	DeepSeekChat = "deepseek-chat"
	// DeepSeek Reasoner (V3) - Chain-of-thought reasoning
	DeepSeekReasoner = "deepseek-reasoner"
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
