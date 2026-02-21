// Example: Basic usage of allm-go with the recommended pattern
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

	// Provider = auth only, Client = behavior
	client := allm.New(
		provider.Anthropic(""), // reads ANTHROPIC_API_KEY from env
		allm.WithModel(provider.AnthropicClaudeSonnet4_5),
		allm.WithMaxTokens(4096),
		allm.WithTemperature(0.7),
		allm.WithTimeout(30*time.Second),
		allm.WithSystemPrompt("You are a helpful assistant. Be concise."),
	)

	// Simple completion
	fmt.Println("=== Simple Completion ===")
	resp, err := client.Complete(ctx, "What is Go in one sentence?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Printf("Model: %s | Tokens: %d in, %d out | Latency: %v\n\n",
		resp.Model, resp.InputTokens, resp.OutputTokens, resp.Latency)

	// Multi-turn conversation
	fmt.Println("=== Multi-turn Chat ===")
	resp, err = client.Chat(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "What is 2+2?"},
		{Role: allm.RoleAssistant, Content: "2+2 equals 4."},
		{Role: allm.RoleUser, Content: "And if you multiply that by 3?"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n\n", resp.Content)

	// Switch model at runtime (same API key, no new provider needed)
	fmt.Println("=== Switch Model ===")
	client.SetModel(provider.AnthropicClaudeHaiku4_5)
	resp, err = client.Complete(ctx, "What is 1+1?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[Haiku] %s (model: %s)\n\n", resp.Content, resp.Model)

	// Streaming
	client.SetModel(provider.AnthropicClaudeSonnet4_5)
	fmt.Println("=== Streaming ===")
	fmt.Print("Response: ")
	for chunk := range client.Stream(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "Count from 1 to 5 slowly."},
	}) {
		if chunk.Error != nil {
			log.Fatal(chunk.Error)
		}
		fmt.Print(chunk.Content)
	}
	fmt.Println()

	// Stream to file
	fmt.Println("\n=== Stream to File ===")
	f, _ := os.CreateTemp("", "allm-*.txt")
	defer os.Remove(f.Name())

	err = client.StreamToWriter(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "Write a haiku about programming."},
	}, f)
	if err != nil {
		log.Fatal(err)
	}
	_ = f.Close()

	content, _ := os.ReadFile(f.Name()) // #nosec G703 -- example code, path is from os.CreateTemp
	fmt.Printf("Written to %s:\n%s\n\n", f.Name(), string(content))

	// Switch to different provider entirely
	fmt.Println("=== Switch Provider ===")
	client.SetProvider(provider.OpenAI(""))
	client.SetModel(provider.OpenAIGPT4o)
	resp, err = client.Complete(ctx, "What provider are you?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[OpenAI] %s\n", resp.Content)
}
