package provider

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/kusandriadi/allm-go"
)

// KeyPattern defines a known API key pattern for a provider.
type KeyPattern struct {
	Provider allm.ProviderName
	Prefix   string         // simple prefix match (e.g. "sk-ant-")
	Regex    *regexp.Regexp // compiled regex for complex patterns
	MinLen   int            // minimum key length
	MaxLen   int            // maximum key length (0 = no limit)
}

// knownKeyPatterns contains patterns for detecting API keys from known providers.
var knownKeyPatterns = []KeyPattern{
	// Anthropic
	{Provider: allm.Anthropic, Prefix: "sk-ant-", MinLen: 40},

	// OpenAI
	{Provider: allm.OpenAI, Prefix: "sk-proj-", MinLen: 40},
	{Provider: allm.OpenAI, Prefix: "sk-svcacct-", MinLen: 40},
	{Provider: allm.OpenAI, Prefix: "sk-", MinLen: 40, MaxLen: 200},

	// DeepSeek
	{Provider: allm.DeepSeek, Prefix: "sk-", MinLen: 30},

	// Groq
	{Provider: allm.Groq, Prefix: "gsk_", MinLen: 30},

	// Moonshot / Kimi
	{Provider: allm.Kimi, Prefix: "sk-", MinLen: 30},

	// Generic bearer-style tokens
	{Provider: "", Regex: regexp.MustCompile(`^eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}`), MinLen: 50},
}

// sourcePatterns are regex patterns for scanning source code for hardcoded keys.
var sourcePatterns = []*regexp.Regexp{
	// Anthropic keys
	regexp.MustCompile(`["']sk-ant-[A-Za-z0-9_-]{20,}["']`),

	// OpenAI keys
	regexp.MustCompile(`["']sk-proj-[A-Za-z0-9_-]{20,}["']`),
	regexp.MustCompile(`["']sk-svcacct-[A-Za-z0-9_-]{20,}["']`),

	// Groq keys
	regexp.MustCompile(`["']gsk_[A-Za-z0-9_-]{20,}["']`),

	// Generic long API keys in quotes (likely hardcoded)
	regexp.MustCompile(`["']sk-[A-Za-z0-9_-]{40,}["']`),

	// Bearer tokens (JWT-like)
	regexp.MustCompile(`["']eyJ[A-Za-z0-9_-]{50,}["']`),

	// Environment variable assignment with actual key
	regexp.MustCompile(`(?:API_KEY|APIKEY|SECRET)\s*[:=]\s*["'][A-Za-z0-9_-]{20,}["']`),
}

// KeyCheckResult describes a detected key issue.
type KeyCheckResult struct {
	Provider allm.ProviderName
	Message  string
	Line     int    // line number (for source scanning)
	File     string // file path (for source scanning)
}

func (r KeyCheckResult) String() string {
	if r.File != "" {
		return fmt.Sprintf("%s:%d: %s", r.File, r.Line, r.Message)
	}
	return r.Message
}

// ValidateKeyFormat checks if an API key matches the expected format for a provider.
// Returns nil if the key looks valid, or an error describing the issue.
func ValidateKeyFormat(provider allm.ProviderName, key string) error {
	if key == "" {
		return nil // empty keys are handled elsewhere (env var fallback)
	}

	switch provider {
	case allm.Anthropic:
		if !strings.HasPrefix(key, "sk-ant-") {
			return fmt.Errorf("anthropic key should start with 'sk-ant-'")
		}
		if len(key) < 40 {
			return fmt.Errorf("anthropic key looks too short (got %d chars)", len(key))
		}

	case allm.OpenAI:
		validPrefixes := []string{"sk-proj-", "sk-svcacct-", "sk-"}
		hasValid := false
		for _, p := range validPrefixes {
			if strings.HasPrefix(key, p) {
				hasValid = true
				break
			}
		}
		if !hasValid {
			return fmt.Errorf("openai key should start with 'sk-'")
		}
		if len(key) < 30 {
			return fmt.Errorf("openai key looks too short (got %d chars)", len(key))
		}

	case allm.Groq:
		if !strings.HasPrefix(key, "gsk_") {
			return fmt.Errorf("groq key should start with 'gsk_'")
		}

	case allm.DeepSeek, allm.Kimi:
		if len(key) < 20 {
			return fmt.Errorf("%s key looks too short (got %d chars)", provider, len(key))
		}

	case allm.Gemini, allm.GLM, allm.Qwen, allm.MiniMax, allm.Perplexity:
		if len(key) < 10 {
			return fmt.Errorf("%s key looks too short (got %d chars)", provider, len(key))
		}
	}

	return nil
}

// DetectKeyInString checks if a string contains what looks like a hardcoded API key.
// Returns the best matching result. Patterns are checked from most specific to least
// specific; once a specific prefix matches (e.g. "sk-ant-"), generic prefixes (e.g. "sk-")
// are skipped.
func DetectKeyInString(s string) []KeyCheckResult {
	var results []KeyCheckResult
	matchedPrefix := ""

	for _, p := range knownKeyPatterns {
		// Skip generic prefix patterns if a more specific one already matched.
		// e.g., if "sk-ant-" matched, skip "sk-" patterns.
		if matchedPrefix != "" && p.Prefix != "" && strings.HasPrefix(matchedPrefix, p.Prefix) && p.Prefix != matchedPrefix {
			continue
		}

		if p.Prefix != "" && strings.Contains(s, p.Prefix) {
			if p.MinLen > 0 && len(s) >= p.MinLen {
				if matchedPrefix == "" || len(p.Prefix) > len(matchedPrefix) {
					matchedPrefix = p.Prefix
				}
				results = append(results, KeyCheckResult{
					Provider: p.Provider,
					Message:  fmt.Sprintf("possible %s API key detected (prefix: %s)", p.Provider, p.Prefix),
				})
				continue
			}
		}
		if p.Regex != nil && p.Regex.MatchString(s) {
			label := string(p.Provider)
			if label == "" {
				label = "unknown provider"
			}
			results = append(results, KeyCheckResult{
				Provider: p.Provider,
				Message:  fmt.Sprintf("possible %s API token detected", label),
			})
		}
	}

	return results
}

// ScanSource scans source code content for hardcoded API keys.
// Pass the file content as a string and the filename for reporting.
// Returns all detected issues.
func ScanSource(content, filename string) []KeyCheckResult {
	var results []KeyCheckResult
	lines := strings.Split(content, "\n")

	for i, line := range lines {
		// Skip comments
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") {
			continue
		}

		found := false
		for _, pat := range sourcePatterns {
			if found {
				break
			}
			if pat.MatchString(line) {
				// Extract the matched key (redacted)
				match := pat.FindString(line)
				redacted := redactKey(match)
				results = append(results, KeyCheckResult{
					File:    filename,
					Line:    i + 1,
					Message: fmt.Sprintf("hardcoded API key detected: %s", redacted),
				})
				found = true
			}
		}
	}

	return results
}

// redactKey partially redacts a key for safe display.
// Shows first 8 and last 4 chars: "sk-ant-abc...wxyz"
func redactKey(key string) string {
	// Remove surrounding quotes
	key = strings.Trim(key, `"'`)

	if len(key) <= 16 {
		return key[:4] + "..." + key[len(key)-2:]
	}
	return key[:8] + "..." + key[len(key)-4:]
}
