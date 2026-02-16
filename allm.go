// Package allm provides a thin, lightweight interface for LLM providers.
//
// Design principles:
//   - Thin: Minimal abstraction over provider APIs
//   - Lightweight: No unnecessary dependencies
//   - Secure: Safe defaults, input validation, context support
//   - Simple: Easy to understand and use
//
// Client is safe for concurrent use. All setter and getter methods
// are protected by a read-write mutex.
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
	"encoding/json"
	"errors"
	"io"
	"sync"
	"time"
)

// Version of the allm-go library
const Version = "0.2.0"

// Common errors
var (
	ErrNoProvider   = errors.New("allm: no provider configured")
	ErrEmptyInput   = errors.New("allm: empty input")
	ErrInputTooLong = errors.New("allm: input exceeds max length")
	ErrRateLimited  = errors.New("allm: rate limited")
	ErrTimeout      = errors.New("allm: request timeout")
	ErrCanceled     = errors.New("allm: request canceled")
	ErrProvider       = errors.New("allm: provider error")
	ErrEmptyResponse  = errors.New("allm: empty response from provider")
)

// Role constants for messages
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Message represents a chat message.
type Message struct {
	Role        string       // "system", "user", "assistant", or "tool"
	Content     string       // Text content
	Images      []Image      // Optional images (for vision models)
	ToolCalls   []ToolCall   // Tool calls requested by assistant (role=assistant)
	ToolResults []ToolResult // Tool results from user (role=tool)
}

// Image represents an image for vision models.
type Image struct {
	MimeType string // e.g., "image/jpeg", "image/png"
	Data     []byte // Raw image bytes (will be base64 encoded)
}

// Tool defines a function that the model can call.
type Tool struct {
	Name        string         // Function name (e.g., "get_weather")
	Description string         // What the function does
	Parameters  map[string]any // JSON Schema for parameters
}

// ToolCall represents a function call requested by the model.
type ToolCall struct {
	ID        string          // Unique call ID (used to match results)
	Name      string          // Function name the model wants to call
	Arguments json.RawMessage // JSON arguments from the model
}

// ToolResult contains the result of a tool call, sent back to the model.
type ToolResult struct {
	ToolCallID string // Must match ToolCall.ID
	Content    string // Result content
	IsError    bool   // True if the tool call failed
}

// RoleTool is the role for messages carrying tool results.
const RoleTool = "tool"

// Request contains parameters for an LLM request.
type Request struct {
	Messages         []Message
	Model            string   // Model to use (empty = provider default)
	MaxTokens        int      // Max tokens to generate (0 = provider default)
	Temperature      float64  // Sampling temperature (0 = provider default)
	TopP             float64  // Nucleus sampling (0 = provider default)
	Stop             []string // Stop sequences
	PresencePenalty  float64  // Presence penalty (-2.0 to 2.0, 0 = default)
	FrequencyPenalty float64  // Frequency penalty (-2.0 to 2.0, 0 = default)
	Tools            []Tool   // Available tools the model can call
}

// Response contains the LLM response.
type Response struct {
	Content      string        // Generated text
	ToolCalls    []ToolCall    // Tool calls requested by the model (when FinishReason is "tool_use" or "tool_calls")
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

// EmbedRequest contains parameters for an embedding request.
type EmbedRequest struct {
	Input []string // Texts to embed
	Model string   // Embedding model (empty = provider default)
}

// EmbedResponse contains the embedding result.
type EmbedResponse struct {
	Embeddings  [][]float64   // One embedding vector per input
	Model       string        // Model used
	Provider    string        // Provider name
	InputTokens int           // Total input tokens
	Latency     time.Duration // Request latency
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

// Embedder is an optional interface providers can implement for text embeddings.
// Supported by: OpenAI, GLM, Local (Ollama/vLLM).
// Not supported by: Anthropic, DeepSeek.
type Embedder interface {
	// Embed generates embeddings for the given texts.
	Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error)
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

// WithPresencePenalty sets the default presence penalty.
// Supported by OpenAI, DeepSeek, and Local providers. Ignored by Anthropic.
func WithPresencePenalty(p float64) Option {
	return func(c *Client) {
		c.presencePenalty = p
	}
}

// WithFrequencyPenalty sets the default frequency penalty.
// Supported by OpenAI, DeepSeek, and Local providers. Ignored by Anthropic.
func WithFrequencyPenalty(p float64) Option {
	return func(c *Client) {
		c.frequencyPenalty = p
	}
}

// WithEmbeddingModel sets the default embedding model.
// This is separate from the chat model set by WithModel.
func WithEmbeddingModel(model string) Option {
	return func(c *Client) {
		c.embeddingModel = model
	}
}

// WithTools sets the tools available for function calling.
func WithTools(tools ...Tool) Option {
	return func(c *Client) {
		c.tools = tools
	}
}

// Client is the main interface for interacting with LLMs.
// It is safe for concurrent use.
type Client struct {
	mu               sync.RWMutex
	provider         Provider
	timeout          time.Duration
	maxInputLen      int
	systemPrompt     string
	model            string  // default chat model
	maxTokens        int     // default max tokens
	temperature      float64 // default temperature
	presencePenalty  float64 // default presence penalty
	frequencyPenalty float64 // default frequency penalty
	embeddingModel   string  // default embedding model
	tools            []Tool  // available tools for function calling
}

// clientState holds a snapshot of client fields for use without holding the lock.
type clientState struct {
	provider         Provider
	timeout          time.Duration
	maxInputLen      int
	systemPrompt     string
	model            string
	maxTokens        int
	temperature      float64
	presencePenalty  float64
	frequencyPenalty float64
	embeddingModel   string
	tools            []Tool
}

// snapshot captures the current client state under a read lock.
func (c *Client) snapshot() clientState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return clientState{
		provider:         c.provider,
		timeout:          c.timeout,
		maxInputLen:      c.maxInputLen,
		systemPrompt:     c.systemPrompt,
		model:            c.model,
		maxTokens:        c.maxTokens,
		temperature:      c.temperature,
		presencePenalty:  c.presencePenalty,
		frequencyPenalty: c.frequencyPenalty,
		embeddingModel:   c.embeddingModel,
		tools:            c.tools,
	}
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

// validateMessages checks message constraints.
func validateMessages(messages []Message, maxInputLen int) error {
	totalLen := 0
	hasContent := false
	for _, m := range messages {
		totalLen += len(m.Content)
		for _, img := range m.Images {
			totalLen += len(img.Data)
		}
		for _, tr := range m.ToolResults {
			totalLen += len(tr.Content)
		}
		if m.Content != "" || len(m.Images) > 0 || len(m.ToolCalls) > 0 || len(m.ToolResults) > 0 {
			hasContent = true
		}
	}
	if !hasContent {
		return ErrEmptyInput
	}
	if totalLen > maxInputLen {
		return ErrInputTooLong
	}
	return nil
}

// buildRequest creates a Request from messages and client state.
func buildRequest(messages []Message, s clientState) *Request {
	msgs := messages
	if s.systemPrompt != "" {
		msgs = append([]Message{{Role: RoleSystem, Content: s.systemPrompt}}, msgs...)
	}
	return &Request{
		Messages:         msgs,
		Model:            s.model,
		MaxTokens:        s.maxTokens,
		Temperature:      s.temperature,
		PresencePenalty:  s.presencePenalty,
		FrequencyPenalty: s.frequencyPenalty,
		Tools:            s.tools,
	}
}

// Complete sends a simple text completion request.
func (c *Client) Complete(ctx context.Context, prompt string) (*Response, error) {
	return c.Chat(ctx, []Message{{Role: RoleUser, Content: prompt}})
}

// Chat sends a multi-turn conversation request.
func (c *Client) Chat(ctx context.Context, messages []Message) (*Response, error) {
	s := c.snapshot()

	if s.provider == nil {
		return nil, ErrNoProvider
	}

	if err := validateMessages(messages, s.maxInputLen); err != nil {
		return nil, err
	}

	req := buildRequest(messages, s)

	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	resp, err := s.provider.Complete(ctx, req)
	if err != nil {
		if errors.Is(err, ErrRateLimited) {
			return nil, err
		}
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

	// Snapshot client state before goroutine to prevent data races.
	s := c.snapshot()

	go func() {
		defer close(out)

		if s.provider == nil {
			out <- StreamChunk{Error: ErrNoProvider}
			return
		}

		if err := validateMessages(messages, s.maxInputLen); err != nil {
			out <- StreamChunk{Error: err}
			return
		}

		req := buildRequest(messages, s)

		ctx, cancel := context.WithTimeout(ctx, s.timeout)
		defer cancel()

		chunks := s.provider.Stream(ctx, req)
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

// Embed generates embeddings for one or more texts.
// Returns an error if the provider does not support embeddings.
func (c *Client) Embed(ctx context.Context, input ...string) (*EmbedResponse, error) {
	s := c.snapshot()

	if s.provider == nil {
		return nil, ErrNoProvider
	}

	embedder, ok := s.provider.(Embedder)
	if !ok {
		return nil, errors.New("allm: provider does not support embeddings")
	}

	if len(input) == 0 {
		return nil, ErrEmptyInput
	}

	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	resp, err := embedder.Embed(ctx, &EmbedRequest{
		Input: input,
		Model: s.embeddingModel,
	})
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

// Models returns available models if the provider supports model listing.
func (c *Client) Models(ctx context.Context) ([]Model, error) {
	c.mu.RLock()
	p := c.provider
	c.mu.RUnlock()

	if p == nil {
		return nil, ErrNoProvider
	}
	lister, ok := p.(ModelLister)
	if !ok {
		return nil, errors.New("allm: provider does not support model listing")
	}
	return lister.Models(ctx)
}

// Provider returns the underlying provider.
func (c *Client) Provider() Provider {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.provider
}

// SetProvider replaces the provider.
func (c *Client) SetProvider(p Provider) {
	c.mu.Lock()
	c.provider = p
	c.mu.Unlock()
}

// SetModel updates the default model.
func (c *Client) SetModel(model string) {
	c.mu.Lock()
	c.model = model
	c.mu.Unlock()
}

// SetSystemPrompt updates the system prompt.
func (c *Client) SetSystemPrompt(prompt string) {
	c.mu.Lock()
	c.systemPrompt = prompt
	c.mu.Unlock()
}

// SetTools updates the available tools for function calling.
func (c *Client) SetTools(tools ...Tool) {
	c.mu.Lock()
	c.tools = tools
	c.mu.Unlock()
}
