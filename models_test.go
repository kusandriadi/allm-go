package allm

import (
	"context"
	"testing"
)

func TestFormatModelListEmpty(t *testing.T) {
	result := FormatModelList(nil)
	if result != "" {
		t.Errorf("expected empty string for nil map, got %q", result)
	}
}

func TestFormatModelListSingle(t *testing.T) {
	models := map[string][]Model{
		"anthropic": {
			{ID: "claude-opus-4-6", Name: "Claude Opus 4.6", Provider: "anthropic"},
		},
	}
	result := FormatModelList(models)
	if !strContains(result, "ANTHROPIC") {
		t.Error("expected provider name in output")
	}
	if !strContains(result, "claude-opus-4-6") {
		t.Error("expected model ID in output")
	}
	if !strContains(result, "1 models") {
		t.Error("expected model count")
	}
}

func TestFormatModelListContextWindow(t *testing.T) {
	models := map[string][]Model{
		"test": {
			{ID: "model-a", ContextWindow: 200000},
		},
	}
	result := FormatModelList(models)
	if !strContains(result, "200k context") {
		t.Errorf("expected context window in output, got %q", result)
	}
}

func TestFormatModelListCompactOver10(t *testing.T) {
	var models []Model
	for i := 0; i < 11; i++ {
		models = append(models, Model{ID: "model-" + string(rune('a'+i)), ContextWindow: 100000})
	}
	result := FormatModelList(map[string][]Model{"test": models})
	if !strContains(result, "100k ctx") {
		t.Errorf("expected compact context format for >10 models, got %q", result)
	}
}

func TestFormatModelListCapabilities(t *testing.T) {
	models := map[string][]Model{
		"test": {
			{ID: "model-a", Capabilities: []string{"vision", "tools"}},
		},
	}
	result := FormatModelList(models)
	if !strContains(result, "[vision, tools]") {
		t.Errorf("expected capabilities in output, got %q", result)
	}
}

func TestFormatModelListSorted(t *testing.T) {
	models := map[string][]Model{
		"openai":    {{ID: "gpt-4"}},
		"anthropic": {{ID: "claude"}},
	}
	result := FormatModelList(models)
	aIdx := strIndex(result, "ANTHROPIC")
	oIdx := strIndex(result, "OPENAI")
	if aIdx > oIdx {
		t.Error("expected providers sorted alphabetically")
	}
}

func TestUsageStatsTracking(t *testing.T) {
	p := &mockProvider{
		name:      "test",
		available: true,
		response: &Response{
			Content:      "hello",
			InputTokens:  10,
			OutputTokens: 5,
		},
	}
	c := New(p)

	usage := c.Usage()
	if usage.Requests != 0 || usage.InputTokens != 0 || usage.OutputTokens != 0 {
		t.Error("expected zero usage before any requests")
	}

	_, err := c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hi"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	usage = c.Usage()
	if usage.Requests != 1 {
		t.Errorf("expected 1 request, got %d", usage.Requests)
	}
	if usage.InputTokens != 10 {
		t.Errorf("expected 10 input tokens, got %d", usage.InputTokens)
	}
	if usage.OutputTokens != 5 {
		t.Errorf("expected 5 output tokens, got %d", usage.OutputTokens)
	}

	// Second request accumulates
	_, _ = c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hi again"}})
	usage = c.Usage()
	if usage.Requests != 2 {
		t.Errorf("expected 2 requests, got %d", usage.Requests)
	}
	if usage.InputTokens != 20 {
		t.Errorf("expected 20 input tokens, got %d", usage.InputTokens)
	}
}

func strContains(s, sub string) bool {
	return strIndex(s, sub) >= 0
}

func strIndex(s, sub string) int {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
