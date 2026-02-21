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

func TestIntegration_Anthropic(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	client := allm.New(
		provider.Anthropic(""),
		allm.WithModel(provider.AnthropicClaudeHaiku3_5),
		allm.WithMaxTokens(256),
	)
	allmtest.Verify(t, client, allmtest.SkipEmbeddings())
}

func TestIntegration_OpenAI(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	client := allm.New(
		provider.OpenAI(""),
		allm.WithModel(provider.OpenAIGPT4oMini),
		allm.WithMaxTokens(256),
	)
	allmtest.Verify(t, client)
}

func TestIntegration_DeepSeek(t *testing.T) {
	if os.Getenv("DEEPSEEK_API_KEY") == "" {
		t.Skip("DEEPSEEK_API_KEY not set")
	}
	client := allm.New(
		provider.DeepSeek(""),
		allm.WithMaxTokens(256),
	)
	allmtest.Verify(t, client,
		allmtest.SkipVision(),
		allmtest.SkipEmbeddings(),
	)
}

func TestIntegration_Gemini(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	client := allm.New(
		provider.Gemini(""),
		allm.WithMaxTokens(256),
	)
	allmtest.Verify(t, client, allmtest.SkipEmbeddings())
}

func TestIntegration_Groq(t *testing.T) {
	if os.Getenv("GROQ_API_KEY") == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	client := allm.New(
		provider.Groq(""),
		allm.WithMaxTokens(256),
	)
	allmtest.Verify(t, client,
		allmtest.SkipVision(),
		allmtest.SkipEmbeddings(),
	)
}

func TestIntegration_GLM(t *testing.T) {
	if os.Getenv("GLM_API_KEY") == "" {
		t.Skip("GLM_API_KEY not set")
	}
	client := allm.New(
		provider.GLM(""),
		allm.WithMaxTokens(256),
	)
	allmtest.Verify(t, client, allmtest.SkipVision())
}

func TestIntegration_Perplexity(t *testing.T) {
	if os.Getenv("PERPLEXITY_API_KEY") == "" {
		t.Skip("PERPLEXITY_API_KEY not set")
	}
	client := allm.New(
		provider.Perplexity(""),
		allm.WithMaxTokens(256),
	)
	allmtest.Verify(t, client,
		allmtest.SkipVision(),
		allmtest.SkipEmbeddings(),
		allmtest.SkipToolUse(),
	)
}
