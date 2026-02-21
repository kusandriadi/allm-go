package provider

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// providerRegistry holds metadata for known OpenAI-compatible providers.
type providerRegistry struct {
	baseURL      string
	defaultModel string
	envKey       string
	embedSupport bool
	// visionSupport indicates the provider has models that support image input.
	// Image data is passed via the shared convertToOpenAI function automatically.
	visionSupport bool
}

// knownProviders is the registry of OpenAI-compatible providers.
var knownProviders = map[allm.ProviderName]providerRegistry{
	allm.DeepSeek: {
		baseURL:      "https://api.deepseek.com/v1",
		defaultModel: "deepseek-chat",
		envKey:       "DEEPSEEK_API_KEY",
		embedSupport: false,
	},
	allm.Gemini: {
		baseURL:       "https://generativelanguage.googleapis.com/v1beta/openai/",
		defaultModel:  "gemini-2.0-flash",
		envKey:        "GEMINI_API_KEY",
		embedSupport:  false,
		visionSupport: true,
	},
	allm.Groq: {
		baseURL:       "https://api.groq.com/openai/v1",
		defaultModel:  "llama-3.3-70b-versatile",
		envKey:        "GROQ_API_KEY",
		embedSupport:  false,
		visionSupport: true,
	},
	allm.GLM: {
		baseURL:       "https://open.bigmodel.cn/api/paas/v4/",
		defaultModel:  "glm-4-flash",
		envKey:        "GLM_API_KEY",
		embedSupport:  true,
		visionSupport: true,
	},
	allm.Perplexity: {
		baseURL:      "https://api.perplexity.ai",
		defaultModel: "llama-3.1-sonar-small-128k-online",
		envKey:       "PERPLEXITY_API_KEY",
		embedSupport: false,
	},
	allm.Local: {
		baseURL:      "http://localhost:11434/v1",
		defaultModel: "llama3",
		envKey:       "LOCAL_API_KEY",
		embedSupport: true,
	},
}

// OpenAICompatibleProvider implements allm.Provider for OpenAI-compatible APIs.
type OpenAICompatibleProvider struct {
	name        allm.ProviderName
	baseURL     string
	apiKey      string
	model       string
	maxTokens   int
	temperature float64
	client      openai.Client
	embedModel  string // for embeddings (GLM, Local)
}

// CompatOption configures the OpenAICompatible provider.
type CompatOption func(*OpenAICompatibleProvider)

// WithBaseURL sets a custom base URL.
func WithBaseURL(url string) CompatOption {
	return func(p *OpenAICompatibleProvider) {
		p.baseURL = url
	}
}

// WithDefaultModel sets the default model.
func WithDefaultModel(model string) CompatOption {
	return func(p *OpenAICompatibleProvider) {
		p.model = model
	}
}

// WithEnvKey sets the environment variable key for the API key.
// This is applied BEFORE the apiKey parameter, so explicit keys override it.
func WithEnvKey(key string) CompatOption {
	return func(p *OpenAICompatibleProvider) {
		if p.apiKey == "" {
			p.apiKey = os.Getenv(key)
		}
	}
}

// WithMaxTokens sets max output tokens.
func WithMaxTokens(n int) CompatOption {
	return func(p *OpenAICompatibleProvider) {
		p.maxTokens = n
	}
}

// WithTemperature sets the temperature.
func WithTemperature(t float64) CompatOption {
	return func(p *OpenAICompatibleProvider) {
		p.temperature = t
	}
}

// WithEmbedModel sets the embedding model (for GLM, Local).
func WithEmbedModel(model string) CompatOption {
	return func(p *OpenAICompatibleProvider) {
		p.embedModel = model
	}
}

// OpenAICompatible creates a new OpenAI-compatible provider.
// If the provider name is in the registry, it uses those defaults.
// Otherwise, it creates a custom provider with the given name.
//
// If apiKey is empty and the provider is in the registry, it reads from the
// provider's default environment variable.
func OpenAICompatible(name allm.ProviderName, apiKey string, opts ...CompatOption) *OpenAICompatibleProvider {
	p := &OpenAICompatibleProvider{
		name:      name,
		baseURL:   "",
		apiKey:    apiKey,
		model:     "",
		maxTokens: 4096,
	}

	// Apply registry defaults if available
	if reg, ok := knownProviders[name]; ok {
		p.baseURL = reg.baseURL
		p.model = reg.defaultModel
		if p.apiKey == "" {
			p.apiKey = os.Getenv(reg.envKey)
		}
		// Set default embedding model for providers that support it
		if reg.embedSupport {
			if name == allm.GLM {
				p.embedModel = "embedding-3"
			} else if name == allm.Local {
				p.embedModel = p.model // Local uses chat model for embeddings by default
			}
		}
	}

	// Apply custom options
	for _, opt := range opts {
		opt(p)
	}

	// Build client
	p.client = p.buildClient()

	return p
}

// Name returns the provider name.
func (p *OpenAICompatibleProvider) Name() string {
	return string(p.name)
}

// Available returns true if the API key is set or if baseURL is set (for local servers without auth).
func (p *OpenAICompatibleProvider) Available() bool {
	// Local providers might not need API key
	if p.name == allm.Local {
		return p.baseURL != ""
	}
	return p.apiKey != ""
}

// Complete sends a completion request.
func (p *OpenAICompatibleProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()

	messages := convertToOpenAI(req.Messages)
	params := openaiChatParams(messages, p.model, p.maxTokens, p.temperature, req)
	model := resolveModel(req.Model, p.model)

	completion, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, wrapOpenAIError(err)
	}

	return openaiCompleteResponse(completion, string(p.name), model, start)
}

// Stream sends a real streaming request using the OpenAI-compatible SDK.
func (p *OpenAICompatibleProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		messages := convertToOpenAI(req.Messages)
		params := openaiChatParams(messages, p.model, p.maxTokens, p.temperature, req)

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)
		openaiStreamLoop(stream, out)
	}()

	return out
}

// Models returns available models from the provider.
func (p *OpenAICompatibleProvider) Models(ctx context.Context) ([]allm.Model, error) {
	return openaiListModels(ctx, p.client, string(p.name))
}

// Embed generates embeddings using the OpenAI-compatible API.
// Supported by providers with embedSupport=true in registry (GLM, Local),
// and by custom (unregistered) providers if an embedding model is configured.
func (p *OpenAICompatibleProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	// Known providers: check registry for embed support
	if reg, ok := knownProviders[p.name]; ok && !reg.embedSupport {
		return nil, fmt.Errorf("allm: provider %s does not support embeddings", p.name)
	}
	// Unknown providers: allow if an embedding model is configured
	if _, ok := knownProviders[p.name]; !ok && p.embedModel == "" && req.Model == "" {
		return nil, fmt.Errorf("allm: provider %s requires an embedding model (use WithEmbedModel)", p.name)
	}

	model := p.embedModel
	if req.Model != "" {
		model = req.Model
	}
	if model == "" {
		model = p.model // Fallback to chat model
	}

	return openaiEmbed(ctx, p.client, req, model, string(p.name))
}

// buildClient creates the OpenAI SDK client with appropriate options.
func (p *OpenAICompatibleProvider) buildClient() openai.Client {
	// Validate base URL for security (allow local URLs for Local provider)
	// Only validate if baseURL looks like a URL (contains scheme or starts with http)
	if p.baseURL != "" && (strings.Contains(p.baseURL, "://") || strings.HasPrefix(p.baseURL, "http")) {
		allowLocal := (p.name == allm.Local)
		if err := validateBaseURLProvider(p.baseURL, allowLocal); err != nil {
			panic(fmt.Sprintf("%s: %v", p.name, err))
		}
	}

	opts := []option.RequestOption{
		option.WithBaseURL(p.baseURL),
	}
	if p.apiKey != "" {
		opts = append(opts, option.WithAPIKey(p.apiKey))
	} else {
		// Fallback to environment variable or dummy key for local servers
		opts = append(opts, option.WithAPIKey("dummy"))
	}
	return openai.NewClient(opts...)
}
