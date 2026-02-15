// Example: List available models from a provider at runtime.
// This lets you discover models dynamically instead of hardcoding.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/kusandriadi/allm-go"
	"github.com/kusandriadi/allm-go/provider"
)

func main() {
	ctx := context.Background()

	// You can list models from any provider.
	// Uncomment the provider you want to try.

	providers := []struct {
		name   string
		client *allm.Client
	}{
		// Reads from ANTHROPIC_API_KEY env
		{"Anthropic", allm.New(provider.Anthropic(""))},
		// Reads from OPENAI_API_KEY env
		{"OpenAI", allm.New(provider.OpenAI(""))},
		// Reads from DEEPSEEK_API_KEY env
		{"DeepSeek", allm.New(provider.DeepSeek(""))},
		// Local Ollama (must be running)
		// {"Local (Ollama)", allm.New(provider.Ollama("llama3"))},
	}

	for _, p := range providers {
		if !p.client.Provider().Available() {
			fmt.Printf("\n--- %s (skipped: no API key) ---\n", p.name)
			continue
		}

		fmt.Printf("\n--- %s models ---\n", p.name)
		models, err := p.client.Models(ctx)
		if err != nil {
			log.Printf("  Error: %v", err)
			continue
		}

		for _, m := range models {
			if m.Name != "" && m.Name != m.ID {
				fmt.Printf("  %s (%s)\n", m.Name, m.ID)
			} else {
				fmt.Printf("  %s\n", m.ID)
			}
		}
		fmt.Printf("  Total: %d models\n", len(models))
	}
}
