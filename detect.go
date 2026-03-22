package allm

import "strings"

// providerPrefixes maps model name substrings to provider names.
var providerPrefixes = map[string]ProviderName{
	"claude":    Anthropic,
	"gpt":       OpenAI,
	"o1":        OpenAI,
	"o3":        OpenAI,
	"o4":        OpenAI,
	"gemini":    Gemini,
	"glm":       GLM,
	"deepseek":  DeepSeek,
	"kimi":      Kimi,
	"moonshot":  Kimi,
	"qwen":      Qwen,
	"minimax":   MiniMax,
	"llama":     Local,
	"mistral":   Local,
	"mixtral":   Local,
	"phi":       Local,
	"codellama": Local,
}

// DetectProvider detects the provider name from a model name by matching
// known substrings. Returns an empty ProviderName if no match is found.
func DetectProvider(model string) ProviderName {
	model = strings.ToLower(model)
	for prefix, prov := range providerPrefixes {
		if strings.Contains(model, prefix) {
			return prov
		}
	}
	return ""
}
