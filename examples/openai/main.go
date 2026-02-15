// Example: OpenAI GPT provider usage
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

	// --- Basic usage (reads OPENAI_API_KEY from env) ---
	client := allm.New(
		provider.OpenAI(""),
		allm.WithTimeout(30*time.Second),
	)

	resp, err := client.Complete(ctx, "What is Go in one sentence?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Printf("Model: %s | Tokens: %d in, %d out | Latency: %v\n\n",
		resp.Model, resp.InputTokens, resp.OutputTokens, resp.Latency)

	// --- Use specific model via constants ---
	clientGPT5 := allm.New(
		provider.OpenAI("",
			provider.WithOpenAIModel(provider.OpenAIGPT5_2),
			provider.WithOpenAIMaxTokens(4096),
		),
	)

	resp, err = clientGPT5.Complete(ctx, "Explain quantum computing in 2 sentences.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[GPT-5.2] %s\n\n", resp.Content)

	// --- Use mini model for fast, cheap responses ---
	clientMini := allm.New(
		provider.OpenAI("",
			provider.WithOpenAIModel(provider.OpenAIGPT4oMini),
		),
	)

	resp, err = clientMini.Complete(ctx, "What is 2+2?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[GPT-4o-mini] %s\n\n", resp.Content)

	// --- Reasoning model (o3) ---
	clientO3 := allm.New(
		provider.OpenAI("",
			provider.WithOpenAIModel(provider.OpenAIO3),
		),
	)

	resp, err = clientO3.Complete(ctx, "How many Rs in the word 'strawberry'?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[o3] %s\n\n", resp.Content)

	// --- Multi-turn conversation ---
	resp, err = client.Chat(ctx, []allm.Message{
		{Role: allm.RoleUser, Content: "My name is Bob."},
		{Role: allm.RoleAssistant, Content: "Hello Bob! How can I help you?"},
		{Role: allm.RoleUser, Content: "What's my name?"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[Chat] %s\n\n", resp.Content)

	// --- System prompt ---
	clientWithSystem := allm.New(
		provider.OpenAI(""),
		allm.WithSystemPrompt("You are a helpful coding assistant. Be concise."),
	)

	resp, err = clientWithSystem.Complete(ctx, "How do I read a file in Go?")
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

	// --- Azure OpenAI ---
	// clientAzure := allm.New(
	//     provider.OpenAI("your-azure-key",
	//         provider.WithOpenAIBaseURL("https://your-resource.openai.azure.com/"),
	//         provider.WithOpenAIModel("your-deployment-name"),
	//     ),
	// )

	// --- List available models ---
	models, err := client.Models(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nAvailable OpenAI models (%d):\n", len(models))
	for _, m := range models {
		fmt.Printf("  - %s\n", m.ID)
	}
}
