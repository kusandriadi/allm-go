package provider

import (
	"testing"
)

func TestAnthropicModelConstants(t *testing.T) {
	models := map[string]string{
		"AnthropicOpus":   AnthropicOpus,
		"AnthropicSonnet": AnthropicSonnet,
		"AnthropicHaiku":  AnthropicHaiku,
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

func TestGLMModelConstants(t *testing.T) {
	models := map[string]string{
		"GLM5Dot1":  GLM5Dot1,
		"GLM5":      GLM5,
		"GLM5Turbo": GLM5Turbo,
		"GLM4Dot7":  GLM4Dot7,
		"GLM4Dot6":  GLM4Dot6,
	}

	for name, val := range models {
		if val == "" {
			t.Errorf("%s should not be empty", name)
		}
	}
}

func TestKimiModelConstants(t *testing.T) {
	models := map[string]string{
		"KimiK2_5":            KimiK2_5,
		"KimiK2Preview":       KimiK2Preview,
		"KimiK2TurboPreview":  KimiK2TurboPreview,
		"KimiK2Thinking":      KimiK2Thinking,
		"KimiK2ThinkingTurbo": KimiK2ThinkingTurbo,
	}
	for name, val := range models {
		if val == "" {
			t.Errorf("%s is empty", name)
		}
	}
}

func TestMiniMaxModelConstants(t *testing.T) {
	models := map[string]string{
		"MiniMaxM2_7":          MiniMaxM2_7,
		"MiniMaxM2_7HighSpeed": MiniMaxM2_7HighSpeed,
		"MiniMaxM2_5":          MiniMaxM2_5,
		"MiniMaxM2_5HighSpeed": MiniMaxM2_5HighSpeed,
		"MiniMaxM2":            MiniMaxM2,
	}
	for name, val := range models {
		if val == "" {
			t.Errorf("%s is empty", name)
		}
	}
}

func TestModelConstantsUsableAsOptions(t *testing.T) {
	// Verify model constants work with provider options
	p1 := Anthropic("key", WithAnthropicModel(AnthropicOpus))
	if p1.model != "opus" {
		t.Errorf("expected opus, got %q", p1.model)
	}

	p2 := OpenAI("key", WithOpenAIModel(OpenAIGPT5_2))
	if p2.model != "gpt-5.2" {
		t.Errorf("expected gpt-5.2, got %q", p2.model)
	}

	p4 := GLM("key", WithAnthropicModel(GLM5))
	if p4.model != "glm-5" {
		t.Errorf("expected glm-5, got %q", p4.model)
	}
}
