// Integration tests for allm-go providers.
// Run with: go test -tags=integration -v
// Requires API keys in environment variables.
//
//go:build integration

package allm_test

import (
	"os"
	"testing"

	"github.com/kusandriadi/allm-go"
	"github.com/kusandriadi/allm-go/allmtest"
	"github.com/kusandriadi/allm-go/provider"
)

// integrationCase defines a provider integration test.
type integrationCase struct {
	name    string
	envKey  string
	setup   func() allm.Provider
	model   string // optional model override
	skips   []allmtest.VerifyOption
}

var integrationCases = []integrationCase{
	{
		name:   "Anthropic",
		envKey: "ANTHROPIC_API_KEY",
		setup:  func() allm.Provider { return provider.Anthropic("") },
		model:  provider.AnthropicClaudeHaiku3_5,
		skips:  []allmtest.VerifyOption{allmtest.SkipEmbeddings()},
	},
	{
		name:   "OpenAI",
		envKey: "OPENAI_API_KEY",
		setup:  func() allm.Provider { return provider.OpenAI("") },
		model:  provider.OpenAIGPT4oMini,
	},
	{
		name:   "DeepSeek",
		envKey: "DEEPSEEK_API_KEY",
		setup:  func() allm.Provider { return provider.DeepSeek("") },
		skips:  []allmtest.VerifyOption{allmtest.SkipVision(), allmtest.SkipEmbeddings()},
	},
	{
		name:   "Gemini",
		envKey: "GEMINI_API_KEY",
		setup:  func() allm.Provider { return provider.Gemini("") },
		skips:  []allmtest.VerifyOption{allmtest.SkipEmbeddings()},
	},
	{
		name:   "GLM",
		envKey: "GLM_API_KEY",
		setup:  func() allm.Provider { return provider.GLM("") },
		skips:  []allmtest.VerifyOption{allmtest.SkipVision()},
	},
	{
		name:   "Kimi",
		envKey: "MOONSHOT_API_KEY",
		setup:  func() allm.Provider { return provider.Kimi("") },
		skips:  []allmtest.VerifyOption{allmtest.SkipVision(), allmtest.SkipEmbeddings()},
	},
	{
		name:   "Qwen",
		envKey: "DASHSCOPE_API_KEY",
		setup:  func() allm.Provider { return provider.Qwen("") },
	},
	{
		name:   "MiniMax",
		envKey: "MINIMAX_API_KEY",
		setup:  func() allm.Provider { return provider.MiniMax("") },
		skips:  []allmtest.VerifyOption{allmtest.SkipVision(), allmtest.SkipEmbeddings()},
	},
}

func TestIntegration(t *testing.T) {
	for _, tc := range integrationCases {
		t.Run(tc.name, func(t *testing.T) {
			if os.Getenv(tc.envKey) == "" {
				t.Skipf("%s not set", tc.envKey)
			}

			opts := []allm.Option{allm.WithMaxTokens(256)}
			if tc.model != "" {
				opts = append(opts, allm.WithModel(tc.model))
			}

			client := allm.New(tc.setup(), opts...)
			allmtest.Verify(t, client, tc.skips...)
		})
	}
}
