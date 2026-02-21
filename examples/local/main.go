// Example: Local provider usage (Ollama, vLLM, etc.)
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

	// --- Ollama (default: localhost:11434) ---
	// Make sure Ollama is running: ollama serve
	// And pull a model first: ollama pull llama3
	clientOllama := allm.New(
		provider.Ollama("llama3"),
		allm.WithTimeout(60*time.Second),
	)

	fmt.Println("=== Ollama ===")
	resp, err := clientOllama.Complete(ctx, "What is Go in one sentence?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Printf("Model: %s | Tokens: %d in, %d out\n\n",
		resp.Model, resp.InputTokens, resp.OutputTokens)

	// --- vLLM (default: localhost:8000) ---
	// Make sure vLLM is running: vllm serve mistralai/Mistral-7B-Instruct-v0.3
	// clientVLLM := allm.New(
	//     provider.VLLM("mistralai/Mistral-7B-Instruct-v0.3"),
	//     allm.WithTimeout(60*time.Second),
	// )

	// --- Custom OpenAI-compatible server ---
	// clientCustom := allm.New(
	//     provider.Local("http://my-server:8080/v1",
	//         provider.WithDefaultModel("my-model"),
	//         provider.WithMaxTokens(2048),
	//         provider.WithTemperature(0.7),
	//     ),
	// )

	// --- Multi-turn conversation ---
	fmt.Println("=== Multi-turn Chat ===")
	resp, err = clientOllama.Chat(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "My name is Charlie."},
		{Role: allm.RoleAssistant, Content: "Hello Charlie!"},
		{Role: allm.RoleUser, Content: "What's my name?"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n\n", resp.Content)

	// --- System prompt ---
	fmt.Println("=== With System Prompt ===")
	clientWithSystem := allm.New(
		provider.Ollama("llama3"),
		allm.WithSystemPrompt("You are a helpful coding assistant. Be very concise."),
	)

	resp, err = clientWithSystem.Complete(ctx, "How do I read a file in Go?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n\n", resp.Content)

	// --- Streaming ---
	fmt.Print("[Stream] ")
	for chunk := range clientOllama.Stream(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "Count from 1 to 5."},
	}) {
		if chunk.Error != nil {
			log.Fatal(chunk.Error)
		}
		fmt.Print(chunk.Content)
	}
	fmt.Println()

	// --- List available models ---
	models, err := clientOllama.Models(ctx)
	if err != nil {
		log.Printf("Model listing error: %v", err)
	} else {
		fmt.Printf("\nAvailable local models (%d):\n", len(models))
		for _, m := range models {
			fmt.Printf("  - %s\n", m.ID)
		}
	}
}
