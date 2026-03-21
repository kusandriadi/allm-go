# AGENTS.md

Guidelines for AI coding agents and contributors working on allm-go.

## Project Structure

```
allm.go              # Core client, types, interfaces
names.go             # ProviderName constants
validation.go        # Input validation, error sanitization
retry.go             # Retry with exponential backoff
provider/
  anthropic.go       # Anthropic Claude (native SDK)
  openai.go          # OpenAI GPT (native SDK)
  claude_cli.go      # Claude CLI provider (exec-based)
  compat.go          # OpenAI-compatible provider + registry
  shortcuts.go       # Provider shortcut constructors
  helpers.go         # Shared helpers, SSRF validation
  keycheck.go        # API key format validation + leak detection
  models.go          # Model name constants
  security_test.go   # OWASP security tests
allmtest/
  allmtest.go         # MockProvider for unit tests
  verify.go          # Verify() for integration tests
integration_test.go  # Table-driven integration tests
examples/            # Usage examples per provider
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

## Testing

```bash
go test ./... -count=1           # unit tests
go test ./... -count=1 -v        # verbose
go test -tags=integration -v     # integration (needs API keys)
go vet ./... && staticcheck ./...  # lint
```

- Tests are **table-driven** where possible (see `compat_test.go`, `integration_test.go`)
- Security tests live in `provider/security_test.go`
- All new features need tests before merge

## Security Rules

This project targets OWASP Top 10 compliance. Non-negotiable:

- **SSRF**: All base URLs validated via `validateBaseURLProvider()` using `net.ParseIP`. Never skip validation.
- **API keys**: Never in error messages, logs, or format strings. Use `sanitizeError()` for logging.
- **CLI injection**: `validateCLIPath()` rejects path traversal and shell metacharacters.
- **Input limits**: `MaxImageSize`, `MaxMaxTokens`, `MaxModelNameLength` etc. in `validation.go`.
- **No `InsecureSkipVerify`**: Never bypass TLS verification.

## Code Style

- No `New` prefix on constructors (e.g., `Anthropic()` not `NewAnthropic()`)
- Provider name constants live in `allm` package (not `provider`) to avoid import cycles
- Rate limiting is the caller's responsibility, not the library's
- Keep dependencies minimal — only 2 SDK deps: `anthropic-sdk-go` + `openai-go`
- Use `panic` only at init/construction time for invalid config (like `regexp.MustCompile`)

## Go Path

```bash
export PATH="$HOME/go-sdk/bin:$HOME/go/bin:$PATH"
```
