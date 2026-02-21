package allm

import (
	"context"
	"time"
)

// retryableOperation is a function that can be retried.
type retryableOperation[T any] func(ctx context.Context) (T, error)

// retryWithBackoff executes an operation with exponential backoff retry logic.
// It handles logging, hooks, error classification, and per-attempt timeouts.
// Used by Chat() and Embed() to avoid code duplication.
func retryWithBackoff[T any](
	ctx context.Context,
	s clientState,
	op retryableOperation[T],
	opName string,
) (T, error) {
	var zero T
	maxAttempts := 1 + s.maxRetries

	if s.hook != nil {
		s.hook(HookEvent{
			Type:     HookRequest,
			Provider: s.provider.Name(),
			Model:    s.model,
			Attempt:  1,
		})
	}

	var lastErr error

	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Backoff before retry
		if attempt > 0 {
			delay := retryDelay(attempt-1, s.retryBaseDelay, s.retryMaxDelay)
			if s.logger != nil {
				s.logger.Warn("retrying "+opName+" request",
					"provider", s.provider.Name(),
					"model", s.model,
					"attempt", attempt+1,
					"delay", delay,
					"error", sanitizeError(lastErr),
				)
			}
			if s.hook != nil {
				s.hook(HookEvent{
					Type:     HookRetry,
					Provider: s.provider.Name(),
					Model:    s.model,
					Attempt:  attempt + 1,
					Error:    sanitizeError(lastErr),
				})
			}
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return zero, ErrCanceled
			}
		}

		// Per-attempt timeout
		attemptCtx, attemptCancel := context.WithTimeout(ctx, s.timeout)
		start := time.Now()
		result, err := op(attemptCtx)
		latency := time.Since(start)

		// Classify error before canceling — attemptCancel sets ctx.Err() to Canceled.
		if err != nil {
			err = classifyError(err, attemptCtx)
		}
		attemptCancel()

		if err == nil {
			if s.logger != nil {
				s.logger.Info(opName+" request succeeded",
					"provider", s.provider.Name(),
					"model", s.model,
					"latency", latency,
					"attempt", attempt+1,
				)
			}
			if s.hook != nil {
				event := HookEvent{
					Type:     HookSuccess,
					Provider: s.provider.Name(),
					Model:    s.model,
					Latency:  latency,
					Attempt:  attempt + 1,
				}
				// Enrich with token counts if result is a *Response
				if resp, ok := any(result).(*Response); ok && resp != nil {
					event.InputTokens = resp.InputTokens
					event.OutputTokens = resp.OutputTokens
				}
				s.hook(event)
			}
			return result, nil
		}

		lastErr = err

		// Don't retry non-transient errors or on the last attempt
		if !isRetryable(lastErr) || attempt == maxAttempts-1 {
			if s.logger != nil {
				s.logger.Error(opName+" request failed",
					"provider", s.provider.Name(),
					"model", s.model,
					"error", sanitizeError(lastErr),
					"attempt", attempt+1,
				)
			}
			if s.hook != nil {
				s.hook(HookEvent{
					Type:     HookError,
					Provider: s.provider.Name(),
					Model:    s.model,
					Latency:  latency,
					Error:    sanitizeError(lastErr),
					Attempt:  attempt + 1,
				})
			}
			return zero, lastErr
		}
	}

	return zero, lastErr
}
