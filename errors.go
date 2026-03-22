package allm

import "errors"

// FormatError returns a user-friendly error message for common LLM errors.
// Sensitive details (API keys, internal paths) are never exposed.
func FormatError(err error) string {
	switch {
	case errors.Is(err, ErrRateLimited):
		return "Too many requests. Please wait a moment."
	case errors.Is(err, ErrInputTooLong):
		return "Message too long. Please shorten it."
	case errors.Is(err, ErrTimeout):
		return "Request timed out. Please try again."
	case errors.Is(err, ErrCanceled):
		return "Request was canceled."
	case errors.Is(err, ErrServerError):
		return "Server error. Please try again later."
	case errors.Is(err, ErrOverloaded):
		return "Provider is overloaded. Please try again later."
	case errors.Is(err, ErrEmptyResponse):
		return "Received an empty response. Please try again."
	case errors.Is(err, ErrNotSupported):
		return "This feature is not supported by the current provider."
	case errors.Is(err, ErrNoProvider):
		return "No AI provider available."
	case errors.Is(err, ErrProvider):
		// Sanitize provider error to avoid leaking sensitive info
		sanitized := sanitizeError(err)
		if sanitized != err {
			return "Provider error. Please try again."
		}
		return "Provider error: " + err.Error()
	case errors.Is(err, ErrEmptyInput):
		return "Empty input provided."
	default:
		return "An error occurred. Please try again."
	}
}
