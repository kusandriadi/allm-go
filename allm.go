// Package allm provides a thin, lightweight interface for LLM providers.
//
// Design principles:
//   - Thin: Minimal abstraction over provider APIs
//   - Lightweight: No unnecessary dependencies
//   - Secure: Safe defaults, input validation, context support
//   - Simple: Easy to understand and use
//
// Basic usage:
//
//	client := allm.New(
//	    provider.Anthropic("sk-ant-..."),
//	)
//	resp, err := client.Complete(ctx, "Hello, world!")
//
// Multi-turn conversation:
//
//	resp, err := client.Chat(ctx, []allm.Message{
//	    {Role: "user", Content: "What is Go?"},
//	    {Role: "assistant", Content: "Go is a programming language..."},
//	    {Role: "user", Content: "Show me an example"},
//	})
package allm

import (
	"context"
	"errors"
	"io"
	"time"
)

// Version of the allm-go library
const Version = "0.1.0"

// Common errors
var (
	ErrNoProvider   = errors.New("allm: no provider configured")
	ErrEmptyInput   = errors.New("allm: empty input")
	ErrInputTooLong = errors.New("allm: input exceeds max length")
	ErrRateLimited  = errors.New("allm: rate limited")
	ErrTimeout      = errors.New("allm: request timeout")
	ErrCanceled     = errors.New("allm: request canceled")
	ErrProvider     = errors.New("allm: provider error")
)

// Role constants for messages
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Message represents a chat message.
type Message struct {
	Role    string  // "system", "user", or "assistant"
	Content string  // Text content
	Images  []Image // Optional images (for vision models)
}

// Image represents an image for vision models.
type Image struct {
	MimeType string // e.g., "image/jpeg", "image/png"
	Data     []byte // Raw image bytes (will be base64 encoded)
}

// Request contains parameters for an LLM request.
type Request struct {
	Messages    []Message
	Model       string   // Model to use (0 = provider default)
	MaxTokens   int      // Max tokens to generate (0 = provider default)
	Temperature float64  // Sampling temperature (0 = provider default)
	TopP        float64  // Nucleus sampling (0 = provider default)
	Stop        []string // Stop sequences
}

// Response contains the LLM response.
type Response struct {
	Content      string        // Generated text
	Provider     string        // Provider name (e.g., "anthropic")
	Model        string        // Model used (e.g., "claude-sonnet-4-20250514")
	InputTokens  int           // Tokens in input
	OutputTokens int           // Tokens in output
	Latency      time.Duration // Request latency
	FinishReason string        // Why generation stopped
}

// StreamChunk represents a chunk of streamed response.
type StreamChunk struct {
	Content string // Partial content
	Done    bool   // True if this is the final chunk
	Error   error  // Non-nil if streaming failed
}

// Provider is the interface that LLM providers must implement.
type Provider interface {
	// Name returns the provider name (e.g., "anthropic", "openai")
	Name() string

	// Complete sends a request and returns the response.
	Complete(ctx context.Context, req *Request) (*Response, error)

	// Stream sends a request and streams the response.
	// Returns a channel that receives chunks until done or error.
	// The channel is closed when streaming completes.
	Stream(ctx context.Context, req *Request) <-chan StreamChunk

	// Available returns true if the provider is properly configured.
	Available() bool
}

// ModelLister is an optional interface providers can implement to list available models.
type ModelLister interface {
	// Models returns the list of available models from the provider.
	Models(ctx context.Context) ([]Model, error)
}

// Model represents an available LLM model.
type Model struct {
	ID       string // Model identifier (e.g., "claude-sonnet-4-20250514")
	Name     string // Human-readable name (e.g., "Claude Sonnet 4")
	Provider string // Provider name
}

// Option configures the Client.
type Option func(*Client)

// WithTimeout sets the request timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		c.timeout = d
	}
}

// WithMaxInputLen sets the maximum input length.
func WithMaxInputLen(n int) Option {
	return func(c *Client) {
		c.maxInputLen = n
	}
}

// WithSystemPrompt sets a system prompt prepended to all requests.
func WithSystemPrompt(prompt string) Option {
	return func(c *Client) {
		c.systemPrompt = prompt
	}
}

// WithModel sets the default model for all requests.
// This overrides the provider's default model.
func WithModel(model string) Option {
	return func(c *Client) {
		c.model = model
	}
}

// WithMaxTokens sets the default max output tokens for all requests.
// This overrides the provider's default max tokens.
func WithMaxTokens(n int) Option {
	return func(c *Client) {
		c.maxTokens = n
	}
}

// WithTemperature sets the default temperature for all requests.
// This overrides the provider's default temperature.
func WithTemperature(t float64) Option {
	return func(c *Client) {
		c.temperature = t
	}
}

// Client is the main interface for interacting with LLMs.
type Client struct {
	provider     Provider
	timeout      time.Duration
	maxInputLen  int
	systemPrompt string
	model        string  // default model (overrides provider default)
	maxTokens    int     // default max tokens (overrides provider default)
	temperature  float64 // default temperature (overrides provider default)
}

// New creates a new Client with the given provider and options.
func New(provider Provider, opts ...Option) *Client {
	c := &Client{
		provider:    provider,
		timeout:     60 * time.Second,
		maxInputLen: 100000, // 100KB default
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Complete sends a simple text completion request.
func (c *Client) Complete(ctx context.Context, prompt string) (*Response, error) {
	return c.Chat(ctx, []Message{{Role: RoleUser, Content: prompt}})
}

// Chat sends a multi-turn conversation request.
func (c *Client) Chat(ctx context.Context, messages []Message) (*Response, error) {
	if c.provider == nil {
		return nil, ErrNoProvider
	}

	// Validate input
	totalLen := 0
	for _, m := range messages {
		totalLen += len(m.Content)
		for _, img := range m.Images {
			totalLen += len(img.Data)
		}
	}
	if totalLen == 0 {
		return nil, ErrEmptyInput
	}
	if totalLen > c.maxInputLen {
		return nil, ErrInputTooLong
	}

	// Build request
	msgs := messages
	if c.systemPrompt != "" {
		msgs = append([]Message{{Role: RoleSystem, Content: c.systemPrompt}}, msgs...)
	}
	req := &Request{
		Messages:    msgs,
		Model:       c.model,
		MaxTokens:   c.maxTokens,
		Temperature: c.temperature,
	}

	// Apply timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Execute
	resp, err := c.provider.Complete(ctx, req)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, ErrTimeout
		}
		if ctx.Err() == context.Canceled {
			return nil, ErrCanceled
		}
		return nil, err
	}
	return resp, nil
}

// Stream sends a request and streams the response.
func (c *Client) Stream(ctx context.Context, messages []Message) <-chan StreamChunk {
	out := make(chan StreamChunk)

	go func() {
		defer close(out)

		if c.provider == nil {
			out <- StreamChunk{Error: ErrNoProvider}
			return
		}

		// Validate input
		totalLen := 0
		for _, m := range messages {
			totalLen += len(m.Content)
		}
		if totalLen == 0 {
			out <- StreamChunk{Error: ErrEmptyInput}
			return
		}
		if totalLen > c.maxInputLen {
			out <- StreamChunk{Error: ErrInputTooLong}
			return
		}

		// Build request
		msgs := messages
		if c.systemPrompt != "" {
			msgs = append([]Message{{Role: RoleSystem, Content: c.systemPrompt}}, msgs...)
		}
		req := &Request{
			Messages:    msgs,
			Model:       c.model,
			MaxTokens:   c.maxTokens,
			Temperature: c.temperature,
		}

		// Apply timeout
		ctx, cancel := context.WithTimeout(ctx, c.timeout)
		defer cancel()

		// Stream from provider
		chunks := c.provider.Stream(ctx, req)
		for chunk := range chunks {
			out <- chunk
			if chunk.Done || chunk.Error != nil {
				return
			}
		}
	}()

	return out
}

// StreamToWriter streams the response directly to an io.Writer.
func (c *Client) StreamToWriter(ctx context.Context, messages []Message, w io.Writer) error {
	for chunk := range c.Stream(ctx, messages) {
		if chunk.Error != nil {
			return chunk.Error
		}
		if _, err := w.Write([]byte(chunk.Content)); err != nil {
			return err
		}
		if chunk.Done {
			return nil
		}
	}
	return nil
}

// Models returns available models if the provider supports model listing.
func (c *Client) Models(ctx context.Context) ([]Model, error) {
	if c.provider == nil {
		return nil, ErrNoProvider
	}
	lister, ok := c.provider.(ModelLister)
	if !ok {
		return nil, errors.New("allm: provider does not support model listing")
	}
	return lister.Models(ctx)
}

// Provider returns the underlying provider.
func (c *Client) Provider() Provider {
	return c.provider
}

// SetProvider replaces the provider.
func (c *Client) SetProvider(p Provider) {
	c.provider = p
}

// SetModel updates the default model.
func (c *Client) SetModel(model string) {
	c.model = model
}

// SetSystemPrompt updates the system prompt.
func (c *Client) SetSystemPrompt(prompt string) {
	c.systemPrompt = prompt
}
