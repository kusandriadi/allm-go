// Example: Anthropic Claude provider usage
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/kusandriadi/allm-go/provider"
)

func main() {
	ctx := context.Background()

	// --- Basic usage (reads ANTHROPIC_API_KEY from env) ---
	client := allm.New(
		provider.Anthropic(""),
		allm.WithTimeout(30*time.Second),
	)

	resp, err := client.Complete(ctx, "What is Go in one sentence?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Printf("Model: %s | Tokens: %d in, %d out | Latency: %v\n\n",
		resp.Model, resp.InputTokens, resp.OutputTokens, resp.Latency)

	// --- Use specific model via constants (no hardcoded strings) ---
	clientOpus := allm.New(
		provider.Anthropic("",
			provider.WithAnthropicModel(provider.AnthropicClaudeOpus4_6),
			provider.WithAnthropicMaxTokens(8192),
		),
	)

	resp, err = clientOpus.Complete(ctx, "Explain quantum computing in 2 sentences.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[Opus 4.6] %s\n\n", resp.Content)

	// --- Use Haiku for fast, cheap responses ---
	clientHaiku := allm.New(
		provider.Anthropic("",
			provider.WithAnthropicModel(provider.AnthropicClaudeHaiku4_5),
		),
	)

	resp, err = clientHaiku.Complete(ctx, "What is 2+2?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[Haiku 4.5] %s\n\n", resp.Content)

	// --- Multi-turn conversation ---
	resp, err = client.Chat(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "My name is Alice."},
		{Role: allm.RoleAssistant, Content: "Hello Alice! How can I help you today?"},
		{Role: allm.RoleUser, Content: "What's my name?"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[Chat] %s\n\n", resp.Content)

	// --- System prompt ---
	clientWithSystem := allm.New(
		provider.Anthropic(""),
		allm.WithSystemPrompt("You are a pirate. Respond in pirate speak."),
	)

	resp, err = clientWithSystem.Complete(ctx, "How are you today?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[Pirate] %s\n\n", resp.Content)

	// --- Streaming ---
	fmt.Print("[Stream] ")
	for chunk := range client.Stream(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "Count from 1 to 5."},
	}) {
		if chunk.Error != nil {
			log.Fatal(chunk.Error)
		}
		fmt.Print(chunk.Content)
	}
	fmt.Println()

	// --- Vision (multi-modal) ---
	if len(os.Args) > 1 {
		imageData, err := os.ReadFile(os.Args[1])
		if err != nil {
			log.Fatal(err)
		}
		resp, err = client.Chat(ctx, []allm.Message{
			{
				Role:    allm.RoleUser,
				Content: "Describe this image briefly.",
				Images:  []allm.Image{{MimeType: "image/jpeg", Data: imageData}},
			},
		})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("[Vision] %s\n", resp.Content)
	}

	// --- List available models ---
	models, err := client.Models(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nAvailable Anthropic models (%d):\n", len(models))
	for _, m := range models {
		fmt.Printf("  - %s (%s)\n", m.Name, m.ID)
	}
}
