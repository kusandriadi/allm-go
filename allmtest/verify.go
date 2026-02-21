// Package allmtest provides testing utilities for allm-go.
//
// # Verify — Feature Test for Library Users
//
// Verify lets users test their provider setup against a real API.
// It checks chat, streaming, vision, embeddings, tool use, and model listing.
//
//	func TestMyProvider(t *testing.T) {
//	    client := allm.New(provider.Anthropic(os.Getenv("ANTHROPIC_API_KEY")))
//	    allmtest.Verify(t, client)
//	}
//
//	// Or pick specific features:
//	allmtest.Verify(t, client,
//	    allmtest.SkipVision(),
//	    allmtest.SkipEmbeddings(),
//	)
package allmtest

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/kusandriadi/allm-go"
)

// VerifyOption configures which features to test.
type VerifyOption func(*verifyConfig)

type verifyConfig struct {
	skipVision     bool
	skipEmbeddings bool
	skipToolUse    bool
	skipStreaming  bool
	skipModels     bool
	timeout        time.Duration
}

// SkipVision skips the vision (image) test.
func SkipVision() VerifyOption {
	return func(c *verifyConfig) { c.skipVision = true }
}

// SkipEmbeddings skips the embeddings test.
func SkipEmbeddings() VerifyOption {
	return func(c *verifyConfig) { c.skipEmbeddings = true }
}

// SkipToolUse skips the tool use / function calling test.
func SkipToolUse() VerifyOption {
	return func(c *verifyConfig) { c.skipToolUse = true }
}

// SkipStreaming skips the streaming test.
func SkipStreaming() VerifyOption {
	return func(c *verifyConfig) { c.skipStreaming = true }
}

// SkipModels skips the model listing test.
func SkipModels() VerifyOption {
	return func(c *verifyConfig) { c.skipModels = true }
}

// WithVerifyTimeout sets the timeout for each test (default: 30s).
func WithVerifyTimeout(d time.Duration) VerifyOption {
	return func(c *verifyConfig) { c.timeout = d }
}

// Verify runs feature tests against a real provider.
// Use in integration tests to validate your provider setup works correctly.
//
//	func TestAnthropic(t *testing.T) {
//	    if os.Getenv("ANTHROPIC_API_KEY") == "" {
//	        t.Skip("ANTHROPIC_API_KEY not set")
//	    }
//	    client := allm.New(provider.Anthropic(""))
//	    allmtest.Verify(t, client)
//	}
func Verify(t *testing.T, client *allm.Client, opts ...VerifyOption) {
	t.Helper()

	cfg := &verifyConfig{
		timeout: 30 * time.Second,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	t.Run("Chat", func(t *testing.T) {
		verifyChat(t, client, cfg.timeout)
	})

	t.Run("MultiTurn", func(t *testing.T) {
		verifyMultiTurn(t, client, cfg.timeout)
	})

	if !cfg.skipStreaming {
		t.Run("Streaming", func(t *testing.T) {
			verifyStreaming(t, client, cfg.timeout)
		})
	}

	if !cfg.skipVision {
		t.Run("Vision", func(t *testing.T) {
			verifyVision(t, client, cfg.timeout)
		})
	}

	if !cfg.skipEmbeddings {
		t.Run("Embeddings", func(t *testing.T) {
			verifyEmbeddings(t, client, cfg.timeout)
		})
	}

	if !cfg.skipToolUse {
		t.Run("ToolUse", func(t *testing.T) {
			verifyToolUse(t, client, cfg.timeout)
		})
	}

	if !cfg.skipModels {
		t.Run("Models", func(t *testing.T) {
			verifyModels(t, client, cfg.timeout)
		})
	}
}

func verifyChat(t *testing.T, client *allm.Client, timeout time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	resp, err := client.Complete(ctx, "Reply with exactly: hello")
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if resp.Content == "" {
		t.Fatal("empty response content")
	}
	if resp.Provider == "" {
		t.Fatal("empty provider name")
	}
	if resp.InputTokens == 0 {
		t.Log("warning: InputTokens is 0")
	}
	if resp.OutputTokens == 0 {
		t.Log("warning: OutputTokens is 0")
	}
	t.Logf("✓ Chat: %q (in:%d out:%d latency:%v)", truncate(resp.Content, 50), resp.InputTokens, resp.OutputTokens, resp.Latency)
}

func verifyMultiTurn(t *testing.T, client *allm.Client, timeout time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	resp, err := client.Chat(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "My name is Alice."},
		{Role: allm.RoleAssistant, Content: "Nice to meet you, Alice!"},
		{Role: allm.RoleUser, Content: "What is my name?"},
	})
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Content), "alice") {
		t.Logf("warning: response may not reference 'Alice': %q", truncate(resp.Content, 100))
	}
	t.Logf("✓ MultiTurn: %q", truncate(resp.Content, 50))
}

func verifyStreaming(t *testing.T, client *allm.Client, timeout time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	chunks := client.Stream(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "Count from 1 to 3."},
	})

	var content string
	var chunkCount int
	for chunk := range chunks {
		if chunk.Error != nil {
			t.Fatalf("Stream error: %v", chunk.Error)
		}
		content += chunk.Content
		chunkCount++
		if chunk.Done {
			break
		}
	}
	if content == "" {
		t.Fatal("empty streaming content")
	}
	t.Logf("✓ Streaming: %d chunks, content: %q", chunkCount, truncate(content, 50))
}

func verifyVision(t *testing.T, client *allm.Client, timeout time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// 1x1 red pixel PNG
	redPixelPNG := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
		0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
		0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
		0x00, 0x00, 0x03, 0x00, 0x01, 0x36, 0x28, 0x19, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
	}

	resp, err := client.Chat(ctx, []allm.Message{
		{
			Role:    allm.RoleUser,
			Content: "What color is this pixel? Reply with just the color name.",
			Images:  []allm.Image{{MimeType: "image/png", Data: redPixelPNG}},
		},
	})
	if err != nil {
		t.Fatalf("Vision failed: %v", err)
	}
	if resp.Content == "" {
		t.Fatal("empty vision response")
	}
	t.Logf("✓ Vision: %q", truncate(resp.Content, 50))
}

func verifyEmbeddings(t *testing.T, client *allm.Client, timeout time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	resp, err := client.Embed(ctx, "Hello world")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(resp.Embeddings) == 0 {
		t.Fatal("no embeddings returned")
	}
	if len(resp.Embeddings[0]) == 0 {
		t.Fatal("empty embedding vector")
	}
	t.Logf("✓ Embeddings: %d dimensions", len(resp.Embeddings[0]))
}

func verifyToolUse(t *testing.T, client *allm.Client, timeout time.Duration) {
	t.Helper()

	// Create a client copy with tools
	toolClient := allm.New(
		client.Provider(),
		allm.WithTools(allm.Tool{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{
						"type":        "string",
						"description": "City name",
					},
				},
				"required": []any{"city"},
			},
		}),
		allm.WithTimeout(timeout),
	)

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	resp, err := toolClient.Complete(ctx, "What is the weather in Tokyo?")
	if err != nil {
		t.Fatalf("ToolUse failed: %v", err)
	}
	if len(resp.ToolCalls) == 0 {
		t.Log("warning: no tool calls returned (model may have answered directly)")
		return
	}
	tc := resp.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Fatalf("expected tool 'get_weather', got %q", tc.Name)
	}
	t.Logf("✓ ToolUse: %s(%s)", tc.Name, string(tc.Arguments))
}

func verifyModels(t *testing.T, client *allm.Client, timeout time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	models, err := client.Models(ctx)
	if err != nil {
		t.Fatalf("Models failed: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("no models returned")
	}
	t.Logf("✓ Models: %d available", len(models))
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
