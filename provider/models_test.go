package provider

import (
	"testing"
)

func TestAnthropicModelConstants(t *testing.T) {
	models := map[string]string{
		"AnthropicClaudeOpus4_6":    AnthropicClaudeOpus4_6,
		"AnthropicClaudeOpus4_5":    AnthropicClaudeOpus4_5,
		"AnthropicClaudeOpus4_1":    AnthropicClaudeOpus4_1,
		"AnthropicClaudeOpus4":      AnthropicClaudeOpus4,
		"AnthropicClaudeSonnet4_5":  AnthropicClaudeSonnet4_5,
		"AnthropicClaudeSonnet4":    AnthropicClaudeSonnet4,
		"AnthropicClaudeSonnet3_7":  AnthropicClaudeSonnet3_7,
		"AnthropicClaudeHaiku4_5":   AnthropicClaudeHaiku4_5,
		"AnthropicClaudeHaiku3_5":   AnthropicClaudeHaiku3_5,
		"AnthropicClaudeHaiku3":     AnthropicClaudeHaiku3,
	}

	for name, val := range models {
		if val == "" {
			t.Errorf("%s should not be empty", name)
		}
	}
}

func TestOpenAIModelConstants(t *testing.T) {
	models := map[string]string{
		"OpenAIGPT5_2":     OpenAIGPT5_2,
		"OpenAIGPT5_2Pro":  OpenAIGPT5_2Pro,
		"OpenAIGPT5_1":     OpenAIGPT5_1,
		"OpenAIGPT5_1Mini": OpenAIGPT5_1Mini,
		"OpenAIGPT5":       OpenAIGPT5,
		"OpenAIGPT5Mini":   OpenAIGPT5Mini,
		"OpenAIGPT5Nano":   OpenAIGPT5Nano,
		"OpenAIGPT4_1":     OpenAIGPT4_1,
		"OpenAIGPT4_1Mini": OpenAIGPT4_1Mini,
		"OpenAIGPT4_1Nano": OpenAIGPT4_1Nano,
		"OpenAIGPT4o":      OpenAIGPT4o,
		"OpenAIGPT4oMini":  OpenAIGPT4oMini,
		"OpenAIGPT4Turbo":  OpenAIGPT4Turbo,
		"OpenAIO4Mini":     OpenAIO4Mini,
		"OpenAIO3":         OpenAIO3,
		"OpenAIO3Mini":     OpenAIO3Mini,
		"OpenAIO1":         OpenAIO1,
	}

	for name, val := range models {
		if val == "" {
			t.Errorf("%s should not be empty", name)
		}
	}
}

func TestDeepSeekModelConstants(t *testing.T) {
	if DeepSeekChat == "" {
		t.Error("DeepSeekChat should not be empty")
	}
	if DeepSeekReasoner == "" {
		t.Error("DeepSeekReasoner should not be empty")
	}
}

func TestModelConstantsUsableAsOptions(t *testing.T) {
	// Verify model constants work with provider options
	p1 := Anthropic("key", WithAnthropicModel(AnthropicClaudeOpus4_6))
	if p1.model != "claude-opus-4-6" {
		t.Errorf("expected claude-opus-4-6, got %q", p1.model)
	}

	p2 := OpenAI("key", WithOpenAIModel(OpenAIGPT5_2))
	if p2.model != "gpt-5.2" {
		t.Errorf("expected gpt-5.2, got %q", p2.model)
	}

	p3 := DeepSeek("key", WithDeepSeekModel(DeepSeekReasoner))
	if p3.model != "deepseek-reasoner" {
		t.Errorf("expected deepseek-reasoner, got %q", p3.model)
	}
}
