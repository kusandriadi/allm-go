package allm

import (
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode"
)

// Input validation constants
const (
	// MaxModelNameLength is the maximum allowed length for model names.
	MaxModelNameLength = 256
	// MaxToolNameLength is the maximum allowed length for tool names.
	MaxToolNameLength = 64
	// MaxStopSequenceLength is the maximum allowed length for stop sequences.
	MaxStopSequenceLength = 128
	// MaxStopSequences is the maximum number of stop sequences allowed.
	MaxStopSequences = 16
	// MaxMaxTokens is the upper bound for max_tokens to prevent misuse.
	MaxMaxTokens = 1_000_000
	// MaxImageSize is the maximum allowed size for a single image (20MB).
	// This prevents OOM from oversized images before the total maxInputLen check.
	MaxImageSize = 20 * 1024 * 1024
	// MaxImageGenPromptLength is the maximum length for image generation prompts.
	MaxImageGenPromptLength = 32_000
	// MaxImageGenCount is the maximum number of images per generation request.
	MaxImageGenCount = 10
	// MaxBatchSize is the maximum number of requests in a batch.
	MaxBatchSize = 50_000
	// MaxCustomIDLength is the maximum length for a batch request custom ID.
	MaxCustomIDLength = 256
)

// Temperature and penalty bounds (OpenAI standard).
const (
	MinTemperature      = 0.0
	MaxTemperature      = 2.0
	MinPresencePenalty  = -2.0
	MaxPresencePenalty  = 2.0
	MinFrequencyPenalty = -2.0
	MaxFrequencyPenalty = 2.0
	MinTopP             = 0.0
	MaxTopP             = 1.0
)

// Retry limits.
const (
	// MinRetryDelay is the minimum backoff delay between retries.
	MinRetryDelay = 1 * time.Millisecond
	// MaxRetryAttempts is the maximum number of retry attempts allowed.
	MaxRetryAttempts = 10
	// MaxRetryMaxDelay is the upper bound for retry max delay.
	MaxRetryMaxDelay = 5 * time.Minute
)

// SanitizeInput removes potentially dangerous control characters from input text.
// It preserves newlines (\n), tabs (\t), and carriage returns (\r).
func SanitizeInput(s string) string {
	s = strings.ReplaceAll(s, "\x00", "")
	return strings.Map(func(r rune) rune {
		if unicode.IsControl(r) && r != '\n' && r != '\t' && r != '\r' {
			return -1
		}
		return r
	}, s)
}

// AllowedImageMIMETypes is the set of accepted image MIME types for vision.
var AllowedImageMIMETypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/gif":  true,
	"image/webp": true,
}

// validateRequest validates request parameters for security and sanity.
func validateRequest(req *Request) error {
	if len(req.Model) > MaxModelNameLength {
		return fmt.Errorf("model name exceeds maximum length of %d", MaxModelNameLength)
	}

	// Temperature: 0 means default, negative is invalid
	if req.Temperature < 0 {
		return fmt.Errorf("temperature must be between %.1f and %.1f", MinTemperature, MaxTemperature)
	}
	if req.Temperature > MaxTemperature {
		return fmt.Errorf("temperature must be between %.1f and %.1f", MinTemperature, MaxTemperature)
	}

	// TopP: 0 means default, negative is invalid
	if req.TopP < 0 {
		return fmt.Errorf("top_p must be between %.1f and %.1f", MinTopP, MaxTopP)
	}
	if req.TopP > MaxTopP {
		return fmt.Errorf("top_p must be between %.1f and %.1f", MinTopP, MaxTopP)
	}

	if req.PresencePenalty < MinPresencePenalty || req.PresencePenalty > MaxPresencePenalty {
		return fmt.Errorf("presence_penalty must be between %.1f and %.1f", MinPresencePenalty, MaxPresencePenalty)
	}
	if req.FrequencyPenalty < MinFrequencyPenalty || req.FrequencyPenalty > MaxFrequencyPenalty {
		return fmt.Errorf("frequency_penalty must be between %.1f and %.1f", MinFrequencyPenalty, MaxFrequencyPenalty)
	}

	if req.MaxTokens < 0 {
		return fmt.Errorf("max_tokens cannot be negative")
	}
	if req.MaxTokens > MaxMaxTokens {
		return fmt.Errorf("max_tokens exceeds maximum of %d", MaxMaxTokens)
	}

	if len(req.Stop) > MaxStopSequences {
		return fmt.Errorf("too many stop sequences (max %d)", MaxStopSequences)
	}
	for i, stop := range req.Stop {
		if len(stop) > MaxStopSequenceLength {
			return fmt.Errorf("stop sequence %d exceeds maximum length of %d", i, MaxStopSequenceLength)
		}
	}

	for i, tool := range req.Tools {
		if tool.Name == "" {
			return fmt.Errorf("tool %d has empty name", i)
		}
		if len(tool.Name) > MaxToolNameLength {
			return fmt.Errorf("tool %d name exceeds maximum length of %d", i, MaxToolNameLength)
		}
	}

	// Validate image MIME types and sizes
	for _, msg := range req.Messages {
		for j, img := range msg.Images {
			if img.MimeType == "" {
				return fmt.Errorf("image %d has empty MIME type", j)
			}
			if !AllowedImageMIMETypes[strings.ToLower(img.MimeType)] {
				return fmt.Errorf("image %d has unsupported MIME type: %s (allowed: jpeg, png, gif, webp)", j, img.MimeType)
			}
			if len(img.Data) == 0 {
				return fmt.Errorf("image %d has empty data", j)
			}
			if len(img.Data) > MaxImageSize {
				return fmt.Errorf("image %d exceeds maximum size of %d bytes (%d bytes)", j, MaxImageSize, len(img.Data))
			}
		}
	}

	// Validate ResponseFormat
	if req.ResponseFormat != nil {
		if req.ResponseFormat.Type != ResponseFormatJSON && req.ResponseFormat.Type != ResponseFormatJSONSchema {
			return fmt.Errorf("invalid response_format type: %s (must be %s or %s)",
				req.ResponseFormat.Type, ResponseFormatJSON, ResponseFormatJSONSchema)
		}
		if req.ResponseFormat.Type == ResponseFormatJSONSchema {
			if req.ResponseFormat.Name == "" {
				return fmt.Errorf("response_format with type json_schema requires a name")
			}
			if req.ResponseFormat.Schema == nil {
				return fmt.Errorf("response_format with type json_schema requires a schema")
			}
		}
	}

	// Validate Thinking
	if req.Thinking != nil {
		if req.Thinking.BudgetTokens < 0 {
			return fmt.Errorf("thinking budget_tokens cannot be negative")
		}
		if req.Thinking.BudgetTokens > MaxMaxTokens {
			return fmt.Errorf("thinking budget_tokens exceeds maximum of %d", MaxMaxTokens)
		}
	}

	// Validate LogProbs
	if req.TopLogProbs < 0 || req.TopLogProbs > 20 {
		return fmt.Errorf("top_log_probs must be between 0 and 20")
	}

	// Validate ParallelToolCalls (only if tools are provided)
	// ParallelToolCalls is allowed even without tools — the provider will ignore it
	// No validation needed here
	_ = req.ParallelToolCalls

	return nil
}

// validateImageRequest validates image generation request parameters.
func validateImageRequest(req *ImageRequest) error {
	if req.Prompt == "" {
		return ErrEmptyInput
	}
	if len(req.Prompt) > MaxImageGenPromptLength {
		return fmt.Errorf("image prompt exceeds maximum length of %d", MaxImageGenPromptLength)
	}
	if req.N < 0 {
		return fmt.Errorf("image count cannot be negative")
	}
	if req.N > MaxImageGenCount {
		return fmt.Errorf("image count exceeds maximum of %d", MaxImageGenCount)
	}
	if req.Model != "" && len(req.Model) > MaxModelNameLength {
		return fmt.Errorf("model name exceeds maximum length of %d", MaxModelNameLength)
	}
	return nil
}

// validateBatchRequests validates batch request parameters.
func validateBatchRequests(requests []BatchRequest) error {
	if len(requests) == 0 {
		return ErrEmptyInput
	}
	if len(requests) > MaxBatchSize {
		return fmt.Errorf("batch size exceeds maximum of %d", MaxBatchSize)
	}
	seen := make(map[string]bool, len(requests))
	for i, r := range requests {
		if r.CustomID == "" {
			return fmt.Errorf("batch request %d has empty custom_id", i)
		}
		if len(r.CustomID) > MaxCustomIDLength {
			return fmt.Errorf("batch request %d custom_id exceeds maximum length of %d", i, MaxCustomIDLength)
		}
		if seen[r.CustomID] {
			return fmt.Errorf("batch request %d has duplicate custom_id: %s", i, r.CustomID)
		}
		seen[r.CustomID] = true
	}
	return nil
}

// sanitizeError removes sensitive information from errors before logging.
// API errors from providers may contain API keys in URLs or headers.
// This wraps non-sentinel errors to prevent accidental exposure.
func sanitizeError(err error) error {
	if err == nil {
		return nil
	}
	// Sentinel errors are safe — return as-is
	if errors.Is(err, ErrRateLimited) || errors.Is(err, ErrServerError) ||
		errors.Is(err, ErrOverloaded) || errors.Is(err, ErrTimeout) ||
		errors.Is(err, ErrCanceled) || errors.Is(err, ErrEmptyResponse) ||
		errors.Is(err, ErrNoProvider) || errors.Is(err, ErrEmptyInput) ||
		errors.Is(err, ErrInputTooLong) || errors.Is(err, ErrProvider) ||
		errors.Is(err, ErrNotSupported) {
		return err
	}
	// Wrap provider errors — expose message but strip potential key material
	msg := err.Error()
	// Check for common patterns that indicate API key leakage
	if containsSensitive(msg) {
		return fmt.Errorf("provider error (details redacted for security)")
	}
	return err
}

// containsSensitive checks if an error message might contain API keys or tokens.
func containsSensitive(msg string) bool {
	lower := strings.ToLower(msg)
	patterns := []string{
		"sk-ant-",        // Anthropic key prefix
		"sk-",            // OpenAI key prefix
		"api_key",        // generic
		"apikey",         // generic
		"bearer ",        // auth header
		"token=",         // token in URL
		"key=",           // key in URL
		"authorization:", // auth header
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}
