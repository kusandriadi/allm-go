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
const Version = "0.8.0"

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
	ErrNotSupported  = errors.New("allm: not supported by provider")
)

// Role constants for messages
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Message represents a chat message.
type Message struct {
	Role         string        // "system", "user", "assistant", or "tool"
	Content      string        // Text content
	Images       []Image       // Optional images (for vision models)
	ToolCalls    []ToolCall    // Tool calls requested by assistant (role=assistant)
	ToolResults  []ToolResult  // Tool results from user (role=tool)
	CacheControl *CacheControl // Prompt caching control (Anthropic)
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

// ResponseFormat constants for structured output.
const (
	ResponseFormatJSON       = "json_object"
	ResponseFormatJSONSchema = "json_schema"
)

// ResponseFormat specifies the format for structured output.
type ResponseFormat struct {
	Type   string         // "json_object" or "json_schema"
	Schema map[string]any // JSON Schema (for json_schema type)
	Name   string         // Schema name (for json_schema type)
}

// CacheControl constants for prompt caching.
const (
	CacheEphemeral = "ephemeral"
)

// CacheControl marks content for prompt caching.
type CacheControl struct {
	Type string // typically "ephemeral"
}

// ThinkingConfig configures extended thinking/reasoning.
type ThinkingConfig struct {
	Type         string // "enabled" for Anthropic
	BudgetTokens int    // token budget for thinking
}

// Truncation strategy constants.
const (
	TruncateTail = "tail" // keep latest messages
	TruncateNone = "none" // error if over limit
)

// TokenCount represents pre-request token counting result.
type TokenCount struct {
	InputTokens int    // Estimated input tokens
	Provider    string // Provider name
	Model       string // Model used
}

// BatchRequest represents a single request in a batch.
type BatchRequest struct {
	CustomID  string    // Custom identifier for this request
	Messages  []Message // Chat messages
	Model     string    // Model to use
	MaxTokens int       // Max tokens to generate
}

// Batch represents a batch processing job.
type Batch struct {
	ID      string        // Batch job ID
	Status  string        // Job status (e.g., "processing", "completed")
	Results []BatchResult // Results (when completed)
}

// BatchResult represents the result of a single batch request.
type BatchResult struct {
	CustomID string    // Matches BatchRequest.CustomID
	Response *Response // Response (nil if error)
	Error    error     // Error (nil if success)
}

// Image size constants for image generation.
const (
	ImageSize256       = "256x256"
	ImageSize512       = "512x512"
	ImageSize1024      = "1024x1024"
	ImageSize1024x1792 = "1024x1792" // Portrait
	ImageSize1792x1024 = "1792x1024" // Landscape
)

// ImageRequest represents an image generation request.
type ImageRequest struct {
	Prompt  string // Text prompt for image generation
	Model   string // Image model (empty = provider default)
	Size    string // Image size (e.g., "1024x1024")
	Quality string // Image quality (e.g., "standard", "hd")
	N       int    // Number of images to generate
}

// GeneratedImage represents a single generated image.
type GeneratedImage struct {
	Data          []byte // Image data (if available)
	URL           string // Image URL (if provider returns URL)
	RevisedPrompt string // Revised prompt (if provider modified it)
}

// ImageResponse represents the result of image generation.
type ImageResponse struct {
	Images   []GeneratedImage // Generated images
	Provider string           // Provider name
	Model    string           // Model used
	Latency  time.Duration    // Request latency
}

// Logger is the interface for structured logging.
// *slog.Logger satisfies this interface out of the box.
type Logger interface {
	Debug(msg string, args ...any)
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
	Model            string          // Model to use (empty = provider default)
	MaxTokens        int             // Max tokens to generate (0 = provider default)
	Temperature      float64         // Sampling temperature (0 = provider default)
	TopP             float64         // Nucleus sampling (0 = provider default)
	Stop             []string        // Stop sequences
	PresencePenalty  float64         // Presence penalty (-2.0 to 2.0, 0 = default)
	FrequencyPenalty float64         // Frequency penalty (-2.0 to 2.0, 0 = default)
	Tools            []Tool          // Available tools the model can call
	ResponseFormat   *ResponseFormat // Structured output format (JSON mode/schema)
	Thinking         *ThinkingConfig // Extended thinking/reasoning config
}

// Response contains the LLM response.
type Response struct {
	Content          string        // Generated text
	ToolCalls        []ToolCall    // Tool calls requested by the model (when FinishReason is "tool_use" or "tool_calls")
	Provider         string        // Provider name (e.g., "anthropic")
	Model            string        // Model used (e.g., "claude-sonnet-4-20250514")
	InputTokens      int           // Tokens in input
	OutputTokens     int           // Tokens in output
	Latency          time.Duration // Request latency
	FinishReason     string        // Why generation stopped
	Thinking         string        // Extended thinking/reasoning content
	ThinkingTokens   int           // Tokens used for thinking
	CacheReadTokens  int           // Tokens read from cache
	CacheWriteTokens int           // Tokens written to cache
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

// TokenCounter is an optional interface for pre-request token counting.
// Supported by: Anthropic (via messages.count_tokens endpoint).
type TokenCounter interface {
	// CountTokens estimates input tokens for a request.
	CountTokens(ctx context.Context, req *Request) (*TokenCount, error)
}

// BatchProvider is an optional interface for batch processing.
// Supported by: OpenAI, Anthropic.
type BatchProvider interface {
	// CreateBatch submits a batch of requests for processing.
	CreateBatch(ctx context.Context, requests []BatchRequest) (*Batch, error)

	// GetBatch retrieves the status and results of a batch job.
	GetBatch(ctx context.Context, batchID string) (*Batch, error)
}

// ImageGenerator is an optional interface for image generation.
// Supported by: OpenAI (DALL-E).
type ImageGenerator interface {
	// GenerateImage creates images from a text prompt.
	GenerateImage(ctx context.Context, req *ImageRequest) (*ImageResponse, error)
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

// WithResponseFormat sets the response format for structured output.
func WithResponseFormat(rf *ResponseFormat) Option {
	return func(c *Client) {
		c.responseFormat = rf
	}
}

// WithThinking enables extended thinking/reasoning with a token budget.
func WithThinking(budgetTokens int) Option {
	return func(c *Client) {
		c.thinking = &ThinkingConfig{
			Type:         "enabled",
			BudgetTokens: budgetTokens,
		}
	}
}

// WithMaxContextTokens sets a soft limit on context tokens.
// Requires provider to support TokenCounter interface.
func WithMaxContextTokens(n int) Option {
	return func(c *Client) {
		c.maxContextTokens = n
	}
}

// WithTruncationStrategy sets how to handle context exceeding MaxContextTokens.
// "tail" keeps latest messages, "none" returns an error.
func WithTruncationStrategy(strategy string) Option {
	return func(c *Client) {
		c.truncationStrategy = strategy
	}
}

// Client is the main interface for interacting with LLMs.
// It is safe for concurrent use.
type Client struct {
	mu                 sync.RWMutex
	provider           Provider
	timeout            time.Duration
	maxInputLen        int
	systemPrompt       string
	model              string          // default chat model
	maxTokens          int             // default max tokens
	temperature        float64         // default temperature
	presencePenalty    float64         // default presence penalty
	frequencyPenalty   float64         // default frequency penalty
	embeddingModel     string          // default embedding model
	tools              []Tool          // available tools for function calling
	maxRetries         int             // 0 = no retry (default)
	retryBaseDelay     time.Duration   // initial backoff delay (default 1s)
	retryMaxDelay      time.Duration   // max backoff delay (default 30s)
	logger             Logger          // structured logger (nil = no logging)
	hook               Hook            // event callback (nil = no hook)
	responseFormat     *ResponseFormat // structured output format
	thinking           *ThinkingConfig // extended thinking config
	maxContextTokens   int             // soft limit on context tokens
	truncationStrategy string          // "tail" or "none"
}

// clientState holds a snapshot of client fields for use without holding the lock.
type clientState struct {
	provider           Provider
	timeout            time.Duration
	maxInputLen        int
	systemPrompt       string
	model              string
	maxTokens          int
	temperature        float64
	presencePenalty    float64
	frequencyPenalty   float64
	embeddingModel     string
	tools              []Tool
	maxRetries         int
	retryBaseDelay     time.Duration
	retryMaxDelay      time.Duration
	logger             Logger
	hook               Hook
	responseFormat     *ResponseFormat
	thinking           *ThinkingConfig
	maxContextTokens   int
	truncationStrategy string
}

// snapshot captures the current client state under a read lock.
func (c *Client) snapshot() clientState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return clientState{
		provider:           c.provider,
		timeout:            c.timeout,
		maxInputLen:        c.maxInputLen,
		systemPrompt:       c.systemPrompt,
		model:              c.model,
		maxTokens:          c.maxTokens,
		temperature:        c.temperature,
		presencePenalty:    c.presencePenalty,
		frequencyPenalty:   c.frequencyPenalty,
		embeddingModel:     c.embeddingModel,
		tools:              c.tools,
		maxRetries:         c.maxRetries,
		retryBaseDelay:     c.retryBaseDelay,
		retryMaxDelay:      c.retryMaxDelay,
		logger:             c.logger,
		hook:               c.hook,
		responseFormat:     c.responseFormat,
		thinking:           c.thinking,
		maxContextTokens:   c.maxContextTokens,
		truncationStrategy: c.truncationStrategy,
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
		ResponseFormat:   s.responseFormat,
		Thinking:         s.thinking,
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

// requestMeta returns safe-to-log metadata about a request.
// Never logs content, images, API keys, or anything sensitive.
func requestMeta(messages []Message, s clientState) []any {
	msgCount := len(messages)
	var imageCount, toolResultCount int
	for _, m := range messages {
		imageCount += len(m.Images)
		toolResultCount += len(m.ToolResults)
	}

	meta := []any{
		"provider", s.provider.Name(),
		"model", s.model,
		"messages", msgCount,
		"timeout", s.timeout,
	}
	if imageCount > 0 {
		meta = append(meta, "images", imageCount)
	}
	if len(s.tools) > 0 {
		meta = append(meta, "tools", len(s.tools))
	}
	if toolResultCount > 0 {
		meta = append(meta, "tool_results", toolResultCount)
	}
	if s.maxTokens > 0 {
		meta = append(meta, "max_tokens", s.maxTokens)
	}
	if s.temperature > 0 {
		meta = append(meta, "temperature", s.temperature)
	}
	if s.systemPrompt != "" {
		meta = append(meta, "has_system_prompt", true)
	}
	if s.maxRetries > 0 {
		meta = append(meta, "max_retries", s.maxRetries)
	}
	if s.responseFormat != nil {
		meta = append(meta, "response_format", s.responseFormat.Type)
	}
	if s.thinking != nil {
		meta = append(meta, "thinking_budget", s.thinking.BudgetTokens)
	}
	if s.maxContextTokens > 0 {
		meta = append(meta, "max_context_tokens", s.maxContextTokens)
	}
	return meta
}

// truncateMessages applies context window management if maxContextTokens is set.
// Returns truncated messages or an error if truncation fails.
func truncateMessages(ctx context.Context, s clientState, messages []Message) ([]Message, error) {
	// Check if provider supports token counting
	counter, ok := s.provider.(TokenCounter)
	if !ok {
		// Provider doesn't support token counting, return messages as-is
		return messages, nil
	}

	// Build a temporary request to count tokens
	tempReq := buildRequest(messages, s)
	count, err := counter.CountTokens(ctx, tempReq)
	if err != nil {
		// If counting fails, return messages as-is (fail open)
		if s.logger != nil {
			s.logger.Debug("token counting failed, skipping truncation", "error", err)
		}
		return messages, nil
	}

	// Check if we're over the limit
	if count.InputTokens <= s.maxContextTokens {
		return messages, nil
	}

	// Over limit - apply truncation strategy
	strategy := s.truncationStrategy
	if strategy == "" {
		strategy = TruncateNone // default to error
	}

	if strategy == TruncateNone {
		return nil, fmt.Errorf("context exceeds max tokens (%d > %d)", count.InputTokens, s.maxContextTokens)
	}

	if strategy == TruncateTail {
		// Keep system messages and remove oldest non-system messages
		var systemMsgs []Message
		var otherMsgs []Message
		for _, m := range messages {
			if m.Role == RoleSystem {
				systemMsgs = append(systemMsgs, m)
			} else {
				otherMsgs = append(otherMsgs, m)
			}
		}

		// Remove oldest non-system messages until under the limit
		for len(otherMsgs) > 1 {
			otherMsgs = otherMsgs[1:]

			// Build a new slice to avoid mutating systemMsgs
			truncated := make([]Message, 0, len(systemMsgs)+len(otherMsgs))
			truncated = append(truncated, systemMsgs...)
			truncated = append(truncated, otherMsgs...)

			tempReq := buildRequest(truncated, s)
			count, err := counter.CountTokens(ctx, tempReq)
			if err != nil {
				return truncated, nil
			}

			if count.InputTokens <= s.maxContextTokens {
				return truncated, nil
			}
		}

		// Return at least system + last message
		result := make([]Message, 0, len(systemMsgs)+len(otherMsgs))
		result = append(result, systemMsgs...)
		result = append(result, otherMsgs...)
		return result, nil
	}

	return nil, fmt.Errorf("unknown truncation strategy: %s", strategy)
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

	if s.logger != nil {
		s.logger.Debug("chat request", requestMeta(messages, s)...)
	}

	if err := validateMessages(messages, s.maxInputLen); err != nil {
		if s.logger != nil {
			s.logger.Debug("chat validation failed", "error", err)
		}
		return nil, err
	}

	// Context window management: truncate if needed
	truncatedMessages := messages
	if s.maxContextTokens > 0 {
		var err error
		truncatedMessages, err = truncateMessages(ctx, s, messages)
		if err != nil {
			if s.logger != nil {
				s.logger.Debug("context truncation failed", "error", err)
			}
			return nil, fmt.Errorf("context truncation: %w", err)
		}
		if len(truncatedMessages) < len(messages) && s.logger != nil {
			s.logger.Debug("context truncated",
				"original_messages", len(messages),
				"truncated_messages", len(truncatedMessages),
			)
		}
	}

	req := buildRequest(truncatedMessages, s)

	// Validate request parameters for security
	if err := validateRequest(req); err != nil {
		if s.logger != nil {
			s.logger.Debug("chat request validation failed", "error", err)
		}
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

		if s.logger != nil {
			s.logger.Debug("stream request", requestMeta(messages, s)...)
		}

		if err := validateMessages(messages, s.maxInputLen); err != nil {
			if s.logger != nil {
				s.logger.Debug("stream validation failed", "error", err)
			}
			out <- StreamChunk{Error: err}
			return
		}

		// Context window management: truncate if needed
		streamMessages := messages
		if s.maxContextTokens > 0 {
			var err error
			streamMessages, err = truncateMessages(ctx, s, messages)
			if err != nil {
				out <- StreamChunk{Error: fmt.Errorf("context truncation: %w", err)}
				return
			}
		}

		req := buildRequest(streamMessages, s)

		// Validate request parameters
		if err := validateRequest(req); err != nil {
			if s.logger != nil {
				s.logger.Debug("stream request validation failed", "error", err)
			}
			out <- StreamChunk{Error: fmt.Errorf("invalid request: %w", err)}
			return
		}

		streamCtx, cancel := context.WithTimeout(ctx, s.timeout)
		defer cancel()

		if s.logger != nil {
			s.logger.Debug("stream starting", "provider", s.provider.Name(), "model", s.model)
		}

		chunks := s.provider.Stream(streamCtx, req)

		var chunkCount int
		// Use select to handle context cancellation properly and prevent goroutine leaks
		for {
			select {
			case <-streamCtx.Done():
				if s.logger != nil {
					s.logger.Debug("stream context done", "provider", s.provider.Name(), "chunks", chunkCount, "error", streamCtx.Err())
				}
				out <- StreamChunk{Error: classifyError(streamCtx.Err(), streamCtx)}
				return
			case chunk, ok := <-chunks:
				if !ok {
					if s.logger != nil {
						s.logger.Debug("stream completed", "provider", s.provider.Name(), "chunks", chunkCount)
					}
					return
				}
				chunkCount++
				out <- chunk
				if chunk.Done {
					if s.logger != nil {
						s.logger.Debug("stream done", "provider", s.provider.Name(), "chunks", chunkCount)
					}
					return
				}
				if chunk.Error != nil {
					if s.logger != nil {
						s.logger.Debug("stream error", "provider", s.provider.Name(), "chunks", chunkCount, "error", sanitizeError(chunk.Error))
					}
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
		return nil, fmt.Errorf("%w: embeddings", ErrNotSupported)
	}

	if len(input) == 0 {
		return nil, ErrEmptyInput
	}

	if s.logger != nil {
		s.logger.Debug("embed request",
			"provider", s.provider.Name(),
			"model", s.embeddingModel,
			"inputs", len(input),
			"timeout", s.timeout,
		)
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
	logger := c.logger
	c.mu.RUnlock()

	if p == nil {
		return nil, ErrNoProvider
	}
	lister, ok := p.(ModelLister)
	if !ok {
		return nil, fmt.Errorf("%w: model listing", ErrNotSupported)
	}

	if logger != nil {
		logger.Debug("models list request", "provider", p.Name())
	}

	models, err := lister.Models(ctx)
	if logger != nil {
		if err != nil {
			logger.Debug("models list failed", "provider", p.Name(), "error", sanitizeError(err))
		} else {
			logger.Debug("models list completed", "provider", p.Name(), "count", len(models))
		}
	}
	return models, err
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

// SetResponseFormat updates the response format for structured output.
// Pass nil to disable structured output.
func (c *Client) SetResponseFormat(rf *ResponseFormat) {
	c.mu.Lock()
	c.responseFormat = rf
	c.mu.Unlock()
}

// SetThinking updates the thinking/reasoning configuration.
// Pass nil to disable extended thinking.
func (c *Client) SetThinking(thinking *ThinkingConfig) {
	c.mu.Lock()
	c.thinking = thinking
	c.mu.Unlock()
}

// CountTokens estimates input tokens for the given messages.
// Returns an error if the provider does not support token counting.
func (c *Client) CountTokens(ctx context.Context, messages []Message) (*TokenCount, error) {
	s := c.snapshot()

	if s.provider == nil {
		return nil, ErrNoProvider
	}

	counter, ok := s.provider.(TokenCounter)
	if !ok {
		return nil, fmt.Errorf("%w: token counting", ErrNotSupported)
	}

	if err := validateMessages(messages, s.maxInputLen); err != nil {
		return nil, err
	}

	req := buildRequest(messages, s)

	if s.logger != nil {
		s.logger.Debug("count tokens request",
			"provider", s.provider.Name(),
			"model", s.model,
			"messages", len(messages),
		)
	}

	return counter.CountTokens(ctx, req)
}

// CreateBatch submits a batch of requests for processing.
// Returns an error if the provider does not support batch processing.
func (c *Client) CreateBatch(ctx context.Context, requests []BatchRequest) (*Batch, error) {
	c.mu.RLock()
	p := c.provider
	logger := c.logger
	c.mu.RUnlock()

	if p == nil {
		return nil, ErrNoProvider
	}

	batcher, ok := p.(BatchProvider)
	if !ok {
		return nil, fmt.Errorf("%w: batch processing", ErrNotSupported)
	}

	if err := validateBatchRequests(requests); err != nil {
		return nil, err
	}

	if logger != nil {
		logger.Debug("create batch request",
			"provider", p.Name(),
			"requests", len(requests),
		)
	}

	return batcher.CreateBatch(ctx, requests)
}

// GetBatch retrieves the status and results of a batch job.
// Returns an error if the provider does not support batch processing.
func (c *Client) GetBatch(ctx context.Context, batchID string) (*Batch, error) {
	c.mu.RLock()
	p := c.provider
	logger := c.logger
	c.mu.RUnlock()

	if p == nil {
		return nil, ErrNoProvider
	}

	batcher, ok := p.(BatchProvider)
	if !ok {
		return nil, fmt.Errorf("%w: batch processing", ErrNotSupported)
	}

	if batchID == "" {
		return nil, ErrEmptyInput
	}

	if logger != nil {
		logger.Debug("get batch request",
			"provider", p.Name(),
			"batch_id", batchID,
		)
	}

	return batcher.GetBatch(ctx, batchID)
}

// ImageOption configures image generation.
type ImageOption func(*ImageRequest)

// WithImageModel sets the image generation model.
func WithImageModel(model string) ImageOption {
	return func(r *ImageRequest) {
		r.Model = model
	}
}

// WithImageSize sets the image size.
func WithImageSize(size string) ImageOption {
	return func(r *ImageRequest) {
		r.Size = size
	}
}

// WithImageQuality sets the image quality.
func WithImageQuality(quality string) ImageOption {
	return func(r *ImageRequest) {
		r.Quality = quality
	}
}

// WithImageCount sets the number of images to generate.
func WithImageCount(n int) ImageOption {
	return func(r *ImageRequest) {
		r.N = n
	}
}

// GenerateImage creates images from a text prompt.
// Returns an error if the provider does not support image generation.
func (c *Client) GenerateImage(ctx context.Context, prompt string, opts ...ImageOption) (*ImageResponse, error) {
	c.mu.RLock()
	p := c.provider
	logger := c.logger
	c.mu.RUnlock()

	if p == nil {
		return nil, ErrNoProvider
	}

	generator, ok := p.(ImageGenerator)
	if !ok {
		return nil, fmt.Errorf("%w: image generation", ErrNotSupported)
	}

	req := &ImageRequest{
		Prompt: prompt,
		N:      1, // default to 1 image
	}
	for _, opt := range opts {
		opt(req)
	}

	if err := validateImageRequest(req); err != nil {
		return nil, err
	}

	if logger != nil {
		logger.Debug("generate image request",
			"provider", p.Name(),
			"model", req.Model,
			"size", req.Size,
			"n", req.N,
		)
	}

	return generator.GenerateImage(ctx, req)
}
