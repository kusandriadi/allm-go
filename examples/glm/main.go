// Example: GLM (Zhipu AI) provider usage
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/kusandriadi/allm-go/provider"
)

func main() {
	ctx := context.Background()

	// --- Basic usage (reads GLM_API_KEY from env) ---
	client := allm.New(
		provider.GLM(""),
		allm.WithTimeout(30*time.Second),
	)

	resp, err := client.Complete(ctx, "What is Go in one sentence?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Printf("Model: %s | Tokens: %d in, %d out | Latency: %v\n\n",
		resp.Model, resp.InputTokens, resp.OutputTokens, resp.Latency)

	// --- Use a specific model ---
	clientGLM5 := allm.New(
		provider.GLM("",
			provider.WithGLMModel(provider.GLM5),
		),
	)

	resp, err = clientGLM5.Complete(ctx, "Write a hello world in Go.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[GLM-5] %s\n\n", resp.Content)

	// --- Multi-turn conversation ---
	resp, err = client.Chat(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "What's the capital of France?"},
		{Role: allm.RoleAssistant, Content: "The capital of France is Paris."},
		{Role: allm.RoleUser, Content: "What's its population?"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[Chat multi-turn] %s\n\n", resp.Content)

	// --- With system prompt ---
	clientWithSystem := allm.New(
		provider.GLM(""),
		allm.WithSystemPrompt("You are a Go programming expert. Answer concisely."),
	)

	resp, err = clientWithSystem.Complete(ctx, "How do I handle errors in Go?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[System] %s\n\n", resp.Content)

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

	// --- Embeddings ---
	embClient := allm.New(
		provider.GLM(""),
		allm.WithEmbeddingModel(provider.GLMEmbedding3),
	)

	embResp, err := embClient.Embed(ctx, "Hello world")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\n[Embedding] dimensions: %d, tokens: %d\n",
		len(embResp.Embeddings[0]), embResp.InputTokens)

	// --- List available models ---
	models, err := client.Models(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nAvailable GLM models (%d):\n", len(models))
	for _, m := range models {
		fmt.Printf("  - %s\n", m.ID)
	}
}
