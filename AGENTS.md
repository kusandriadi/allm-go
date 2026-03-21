# AGENTS.md

Guidelines for AI coding agents and contributors working on allm-go.

## Project Structure

```
allm.go              # Core client, types, interfaces, options
names.go             # ProviderName constants
validation.go        # Input validation, error sanitization
retry.go             # Retry with exponential backoff
features_test.go     # Tests for v0.8.0 features (structured output, thinking, etc.)
provider/
  anthropic.go       # Anthropic Claude (native SDK) — thinking, caching, token counting
  openai.go          # OpenAI GPT (native SDK) — image generation, batch stubs
  claude_cli.go      # Claude CLI provider (exec-based)
  compat.go          # OpenAI-compatible provider + registry
  shortcuts.go       # Provider shortcut constructors
  helpers.go         # Shared helpers: message conversion, response format, SSRF validation
  keycheck.go        # API key format validation + leak detection
  models.go          # Model name constants
  security_test.go   # OWASP security tests
allmtest/
  allmtest.go         # MockProvider for unit tests
  verify.go          # Verify() for integration tests
integration_test.go  # Table-driven integration tests
examples/            # Usage examples per provider
```

## Key Interfaces

```go
Provider       // Core: Name, Complete, Stream, Available
ModelLister    // Optional: Models listing
Embedder       // Optional: text embeddings
TokenCounter   // Optional: pre-request token counting (Anthropic)
BatchProvider  // Optional: batch processing
ImageGenerator // Optional: image generation (OpenAI)
```

## Adding a New Provider

If OpenAI-compatible (most are):

1. Add `ProviderName` constant in `names.go`
2. Add registry entry in `provider/compat.go` (`knownProviders` map)
3. Add shortcut function in `provider/shortcuts.go`
4. Add model constants in `provider/models.go`
5. Add key validation in `provider/keycheck.go` (if provider has a known key prefix)
6. Add integration test case in `integration_test.go`
7. Update `README.md` feature matrix

No new files needed — the `OpenAICompatibleProvider` handles everything.

## Adding a New Feature

1. Add types/interfaces to `allm.go`
2. Add `With*` option and field to `Client`, `clientState`, `snapshot()`
3. Add `Set*` runtime setter for consistency
4. Add validation in `validation.go`
5. Implement in relevant providers
6. Add to `buildRequest()` if it's a request parameter
7. Add debug logging (safe metadata only, never content/keys)
8. Add tests in `features_test.go` or relevant `*_test.go`
9. Update `README.md` and this file

## Testing

```bash
go test ./... -count=1           # unit tests
go test ./... -count=1 -v        # verbose
go test -tags=integration -v     # integration (needs API keys)
go vet ./... && staticcheck ./...  # lint
```

- Tests are **table-driven** where possible (see `compat_test.go`, `integration_test.go`)
- Security tests live in `provider/security_test.go`
- Feature tests in `features_test.go`
- All new features need tests before merge

## Security Rules

This project targets OWASP Top 10 compliance. Non-negotiable:

- **SSRF (A10)**: All base URLs validated via `validateBaseURLProvider()` using `net.ParseIP`. Never skip validation.
- **API keys (A07)**: Never in error messages, logs, or format strings. Use `sanitizeError()` / `sanitizeProviderError()` for logging.
- **CLI injection (A03)**: `validateCLIPath()` rejects path traversal and shell metacharacters.
- **Input limits (A05)**: `MaxImageSize`, `MaxMaxTokens`, `MaxModelNameLength`, `MaxImageGenPromptLength`, `MaxBatchSize`, etc. in `validation.go`.
- **No `InsecureSkipVerify`**: Never bypass TLS verification.
- **Error sentinel**: Use `ErrNotSupported` for unsupported features, not format strings.

## Code Style

- No `New` prefix on constructors (e.g., `Anthropic()` not `NewAnthropic()`)
- Provider name constants live in `allm` package (not `provider`) to avoid import cycles
- Rate limiting is the caller's responsibility, not the library's
- Keep dependencies minimal — only 2 SDK deps: `anthropic-sdk-go` + `openai-go`
- Use `panic` only at init/construction time for invalid config (like `regexp.MustCompile`)
- Use `ErrNotSupported` sentinel for all "provider does not support X" errors
- All `Client` fields must be in `clientState` + `snapshot()` for thread safety
- Every `With*` option should have a corresponding `Set*` runtime setter

## Go Path

```bash
export PATH="$HOME/go-sdk/bin:$HOME/go/bin:$PATH"
```
