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
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"sync"
	"time"
)

// Version of the allm-go library
const Version = "0.5.0"

// Common errors
var (
	ErrNoProvider    = errors.New("allm: no provider configured")
	ErrEmptyInput    = errors.New("allm: empty input")
	ErrInputTooLong  = errors.New("allm: input exceeds max length")
	ErrRateLimited   = errors.New("allm: rate limited")
	ErrTimeout       = errors.New("allm: request timeout")
	ErrCanceled      = errors.New("allm: request canceled")
	ErrProvider      = errors.New("allm: provider error")
	ErrEmptyResponse = errors.New("allm: empty response from provider")
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

// Logger is the interface for structured logging.
// *slog.Logger satisfies this interface out of the box.
type Logger interface {
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
}

// Hook event type constants.
const (
	HookRequest = "request"
	HookSuccess = "success"
	HookError   = "error"
	HookRetry   = "retry"
)

// HookEvent contains information about a client event.
type HookEvent struct {
	Type         string        // HookRequest, HookSuccess, HookError, HookRetry
	Provider     string        // Provider name
	Model        string        // Model used
	Latency      time.Duration // Request latency
	InputTokens  int           // Input token count
	OutputTokens int           // Output token count
	Error        error         // Error, if any
	Attempt      int           // Current attempt (1-based)
}

// Hook is a callback for observing client events.
type Hook func(event HookEvent)

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

// WithMaxRetries sets the maximum number of retry attempts for transient errors.
// Default is 0 (no retries). Retries use exponential backoff.
// Maximum allowed value is 10. Panics if n is invalid.
func WithMaxRetries(n int) Option {
	return func(c *Client) {
		if n < 0 || n > 10 {
			panic(fmt.Sprintf("invalid max_retries: %d (must be 0-10)", n))
		}
		c.maxRetries = n
	}
}

// WithRetryBaseDelay sets the initial backoff delay between retries.
// Default is 1 second. Minimum allowed is 100ms. Panics if invalid.
func WithRetryBaseDelay(d time.Duration) Option {
	return func(c *Client) {
		if d < MinRetryDelay {
			panic(fmt.Sprintf("invalid retry_base_delay: %v (minimum %v)", d, MinRetryDelay))
		}
		c.retryBaseDelay = d
	}
}

// WithRetryMaxDelay sets the maximum backoff delay between retries.
// Default is 30 seconds. Must be >= MinRetryDelay and <= 5 minutes. Panics if invalid.
// Note: maxDelay >= baseDelay is NOT validated here because option order is arbitrary.
// If maxDelay < baseDelay at runtime, retryDelay clamps to maxDelay.
func WithRetryMaxDelay(d time.Duration) Option {
	return func(c *Client) {
		if d < MinRetryDelay {
			panic(fmt.Sprintf("invalid retry_max_delay: %v (minimum %v)", d, MinRetryDelay))
		}
		if d > MaxRetryMaxDelay {
			panic(fmt.Sprintf("invalid retry_max_delay: %v (maximum %v)", d, MaxRetryMaxDelay))
		}
		c.retryMaxDelay = d
	}
}

// WithLogger sets a structured logger for the client.
// Pass slog.Default() for standard logging, or nil to disable (default).
func WithLogger(logger Logger) Option {
	return func(c *Client) {
		c.logger = logger
	}
}

// WithHook sets a callback for observing client events (requests, retries, errors).
func WithHook(hook Hook) Option {
	return func(c *Client) {
		c.hook = hook
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
	model            string        // default chat model
	maxTokens        int           // default max tokens
	temperature      float64       // default temperature
	presencePenalty  float64       // default presence penalty
	frequencyPenalty float64       // default frequency penalty
	embeddingModel   string        // default embedding model
	tools            []Tool        // available tools for function calling
	maxRetries       int           // 0 = no retry (default)
	retryBaseDelay   time.Duration // initial backoff delay (default 1s)
	retryMaxDelay    time.Duration // max backoff delay (default 30s)
	logger           Logger        // structured logger (nil = no logging)
	hook             Hook          // event callback (nil = no hook)
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
	maxRetries       int
	retryBaseDelay   time.Duration
	retryMaxDelay    time.Duration
	logger           Logger
	hook             Hook
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
		maxRetries:       c.maxRetries,
		retryBaseDelay:   c.retryBaseDelay,
		retryMaxDelay:    c.retryMaxDelay,
		logger:           c.logger,
		hook:             c.hook,
	}
}

// New creates a new Client with the given provider and options.
func New(provider Provider, opts ...Option) *Client {
	c := &Client{
		provider:       provider,
		timeout:        60 * time.Second,
		maxInputLen:    100000, // 100KB default
		retryBaseDelay: 1 * time.Second,
		retryMaxDelay:  30 * time.Second,
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

// isRetryable returns true if the error is transient and worth retrying.
func isRetryable(err error) bool {
	if errors.Is(err, ErrRateLimited) || errors.Is(err, ErrTimeout) || errors.Is(err, ErrEmptyResponse) {
		return true
	}
	return false
}

// retryDelay calculates the backoff delay for a given attempt using exponential backoff with jitter.
// attempt is 0-based (0 = first retry).
func retryDelay(attempt int, base, max time.Duration) time.Duration {
	delay := time.Duration(float64(base) * math.Pow(2, float64(attempt)))
	if delay > max {
		delay = max
	}
	// Add 0-25% jitter (non-cryptographic, safe for backoff timing)
	jitter := time.Duration(float64(delay) * 0.25 * rand.Float64()) // #nosec G404 -- jitter does not need crypto rand
	return delay + jitter
}

// classifyError converts context errors to allm sentinel errors.
func classifyError(err error, ctx context.Context) error {
	if errors.Is(err, ErrRateLimited) || errors.Is(err, ErrEmptyResponse) || errors.Is(err, ErrTimeout) {
		return err
	}
	if ctx.Err() == context.DeadlineExceeded {
		return ErrTimeout
	}
	if ctx.Err() == context.Canceled {
		return ErrCanceled
	}
	return err
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

	// Validate request parameters for security
	if err := validateRequest(req); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}

	return retryWithBackoff(ctx, s, func(attemptCtx context.Context) (*Response, error) {
		return s.provider.Complete(attemptCtx, req)
	}, "chat")
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

		// Validate request parameters
		if err := validateRequest(req); err != nil {
			out <- StreamChunk{Error: fmt.Errorf("invalid request: %w", err)}
			return
		}

		streamCtx, cancel := context.WithTimeout(ctx, s.timeout)
		defer cancel()

		chunks := s.provider.Stream(streamCtx, req)

		// Use select to handle context cancellation properly and prevent goroutine leaks
		for {
			select {
			case <-streamCtx.Done():
				out <- StreamChunk{Error: classifyError(streamCtx.Err(), streamCtx)}
				return
			case chunk, ok := <-chunks:
				if !ok {
					// Provider closed the channel without sending Done - this is valid
					return
				}
				out <- chunk
				if chunk.Done || chunk.Error != nil {
					return
				}
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
		return nil, fmt.Errorf("allm: provider does not support embeddings")
	}

	if len(input) == 0 {
		return nil, ErrEmptyInput
	}

	// Validate input strings (check for excessive length)
	for i, text := range input {
		if len(text) > s.maxInputLen {
			return nil, fmt.Errorf("input %d exceeds max length", i)
		}
	}

	embedReq := &EmbedRequest{
		Input: input,
		Model: s.embeddingModel,
	}

	// Use generic retry helper
	return retryWithBackoff(ctx, s, func(attemptCtx context.Context) (*EmbedResponse, error) {
		return embedder.Embed(attemptCtx, embedReq)
	}, "embed")
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
