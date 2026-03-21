package provider

import (
	"strings"
	"testing"

	"github.com/kusandriadi/allm-go"
)

func TestValidateKeyFormat(t *testing.T) {
	tests := []struct {
		name     string
		provider allm.ProviderName
		key      string
		wantErr  bool
	}{
		// Empty key (ok - handled by env fallback)
		{"empty key", allm.Anthropic, "", false},

		// Anthropic
		{"valid anthropic", allm.Anthropic, "sk-ant-api03-" + strings.Repeat("x", 40), false},
		{"bad anthropic prefix", allm.Anthropic, "sk-bad-" + strings.Repeat("x", 40), true},
		{"short anthropic", allm.Anthropic, "sk-ant-short", true},

		// OpenAI
		{"valid openai proj", allm.OpenAI, "sk-proj-" + strings.Repeat("x", 40), false},
		{"valid openai svc", allm.OpenAI, "sk-svcacct-" + strings.Repeat("x", 40), false},
		{"valid openai sk", allm.OpenAI, "sk-" + strings.Repeat("x", 40), false},
		{"bad openai prefix", allm.OpenAI, "bad-" + strings.Repeat("x", 40), true},
		{"short openai", allm.OpenAI, "sk-short", true},

		// DeepSeek
		{"valid deepseek", allm.DeepSeek, strings.Repeat("x", 30), false},
		{"short deepseek", allm.DeepSeek, "short", true},

		// Kimi
		{"valid kimi", allm.Kimi, strings.Repeat("x", 30), false},
		{"short kimi", allm.Kimi, "short", true},

		// Qwen
		{"valid qwen", allm.Qwen, strings.Repeat("x", 20), false},
		{"short qwen", allm.Qwen, "tiny", true},

		// MiniMax
		{"valid minimax", allm.MiniMax, strings.Repeat("x", 20), false},
		{"short minimax", allm.MiniMax, "tiny", true},

		// Gemini
		{"valid gemini", allm.Gemini, strings.Repeat("x", 20), false},
		{"short gemini", allm.Gemini, "tiny", true},

		// GLM
		{"valid glm", allm.GLM, strings.Repeat("x", 20), false},
		{"short glm", allm.GLM, "tiny", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateKeyFormat(tt.provider, tt.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateKeyFormat(%s, %q) error = %v, wantErr %v", tt.provider, truncate(tt.key), err, tt.wantErr)
			}
		})
	}
}

func TestDetectKeyInString(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantCount int
	}{
		{"no key", "hello world", 0},
		{"anthropic key", "sk-ant-api03-" + strings.Repeat("a", 80), 1},
		{"openai key", "sk-proj-" + strings.Repeat("b", 80), 1},
		{"short string", "sk-ant-short", 0},
		{"jwt token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := DetectKeyInString(tt.input)
			if len(results) != tt.wantCount {
				t.Errorf("DetectKeyInString(%q) got %d results, want %d", truncate(tt.input), len(results), tt.wantCount)
			}
		})
	}
}

func TestScanSource(t *testing.T) {
	tests := []struct {
		name      string
		content   string
		wantCount int
	}{
		{
			"clean code",
			`p := provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY"))`,
			0,
		},
		{
			"hardcoded anthropic key",
			`p := provider.Anthropic("sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")`,
			1,
		},
		{
			"hardcoded openai key",
			`client := openai.New("sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")`,
			1,
		},
		{
			"env var assignment with key",
			`API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`,
			1,
		},
		{
			"comment ignored",
			`// sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`,
			0,
		},
		{
			"multiple keys",
			"line1 := \"sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"\n" +
				"line2 := \"sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"\n",
			2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := ScanSource(tt.content, "test.go")
			if len(results) != tt.wantCount {
				t.Errorf("ScanSource() got %d results, want %d", len(results), tt.wantCount)
				for _, r := range results {
					t.Logf("  %s", r)
				}
			}
		})
	}
}

func TestScanSourceLineNumbers(t *testing.T) {
	content := "line1\n" +
		"line2\n" +
		`key := "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"` + "\n" +
		"line4\n"

	results := ScanSource(content, "main.go")
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Line != 3 {
		t.Errorf("expected line 3, got %d", results[0].Line)
	}
	if results[0].File != "main.go" {
		t.Errorf("expected file main.go, got %q", results[0].File)
	}
}

func TestRedactKey(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{`"sk-ant-api03-abcdefghijklmnop"`, "sk-ant-a...mnop"},
		{`"short"`, "shor...rt"},
	}

	for _, tt := range tests {
		got := redactKey(tt.input)
		if got != tt.want {
			t.Errorf("redactKey(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestKeyCheckResultString(t *testing.T) {
	// With file
	r1 := KeyCheckResult{File: "main.go", Line: 42, Message: "key found"}
	if s := r1.String(); s != "main.go:42: key found" {
		t.Errorf("unexpected: %q", s)
	}

	// Without file
	r2 := KeyCheckResult{Message: "key found"}
	if s := r2.String(); s != "key found" {
		t.Errorf("unexpected: %q", s)
	}
}

func truncate(s string) string {
	if len(s) > 30 {
		return s[:30] + "..."
	}
	return s
}
