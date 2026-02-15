// Example: Basic usage of allm-go
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
	// Create a client with Anthropic provider
	client := allm.New(
		provider.Anthropic(""), // Empty = reads from ANTHROPIC_API_KEY env
		allm.WithTimeout(30*time.Second),
		allm.WithSystemPrompt("You are a helpful assistant. Be concise."),
	)

	ctx := context.Background()

	// Simple completion
	fmt.Println("=== Simple Completion ===")
	resp, err := client.Complete(ctx, "What is Go in one sentence?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Printf("Tokens: %d in, %d out\n", resp.InputTokens, resp.OutputTokens)
	fmt.Printf("Latency: %v\n\n", resp.Latency)

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

	// Streaming
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
	fmt.Println("=== Stream to File ===")
	f, _ := os.CreateTemp("", "allm-*.txt")
	defer os.Remove(f.Name())
	
	err = client.StreamToWriter(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "Write a haiku about programming."},
	}, f)
	if err != nil {
		log.Fatal(err)
	}
	f.Close()
	
	content, _ := os.ReadFile(f.Name())
	fmt.Printf("Written to %s:\n%s\n", f.Name(), string(content))
}
