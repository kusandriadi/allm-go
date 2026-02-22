// Claude CLI provider — uses the `claude` command-line tool for completions.
// Designed for Claude Pro/Max subscribers who authenticate via Claude Code OAuth.
// No API key needed; authentication is handled by the CLI itself.
package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/kusandriadi/allm-go"
)

// ClaudeCLIProvider implements allm.Provider using the claude CLI.
type ClaudeCLIProvider struct {
	model   string
	cliPath string // path to claude binary (default: "claude")
	effort  string // effort level: low, medium, high (optional)
}

// CLIOption configures the ClaudeCLI provider.
type CLIOption func(*ClaudeCLIProvider)

// WithCLIModel sets the model.
func WithCLIModel(model string) CLIOption {
	return func(p *ClaudeCLIProvider) {
		p.model = model
	}
}

// WithCLIPath sets a custom path to the claude binary.
func WithCLIPath(path string) CLIOption {
	return func(p *ClaudeCLIProvider) {
		p.cliPath = path
	}
}

// WithCLIEffort sets the effort level (low, medium, high).
func WithCLIEffort(effort string) CLIOption {
	return func(p *ClaudeCLIProvider) {
		p.effort = effort
	}
}

// ClaudeCLI creates a new Claude CLI provider.
func ClaudeCLI(opts ...CLIOption) *ClaudeCLIProvider {
	p := &ClaudeCLIProvider{
		model:   "claude-sonnet-4-20250514",
		cliPath: "claude",
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// Name returns the provider name.
func (p *ClaudeCLIProvider) Name() string {
	return "claude-cli"
}

// Available returns true if the claude binary is found in PATH.
func (p *ClaudeCLIProvider) Available() bool {
	_, err := exec.LookPath(p.cliPath)
	return err == nil
}

// cliResult represents the JSON output from claude CLI (--output-format json).
type cliResult struct {
	Type       string  `json:"type"`
	Subtype    string  `json:"subtype"`
	IsError    bool    `json:"is_error"`
	Result     string  `json:"result"`
	DurationMs int     `json:"duration_ms"`
	TotalCost  float64 `json:"total_cost_usd"`
	Usage      struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

// cliStreamMessage represents an assistant message in stream-json output.
type cliStreamMessage struct {
	Type    string `json:"type"`
	Message struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"message"`
}

// buildArgs constructs claude CLI arguments from a request.
func (p *ClaudeCLIProvider) buildArgs(req *allm.Request, outputFormat string) (args []string, prompt string) {
	args = []string{"-p", "--output-format", outputFormat, "--no-session-persistence"}

	if outputFormat == "stream-json" {
		args = append(args, "--verbose", "--include-partial-messages")
	}

	// Model
	model := p.model
	if req.Model != "" {
		model = req.Model
	}
	args = append(args, "--model", model)

	// Effort
	if p.effort != "" {
		args = append(args, "--effort", p.effort)
	}

	// Disable tools — pure LLM mode
	args = append(args, "--tools", "")

	// Extract system prompt and build conversation prompt
	var systemParts []string
	var convParts []string

	for _, m := range req.Messages {
		switch m.Role {
		case allm.RoleSystem:
			systemParts = append(systemParts, m.Content)
		case allm.RoleUser:
			convParts = append(convParts, "Human: "+m.Content)
		case allm.RoleAssistant:
			convParts = append(convParts, "Assistant: "+m.Content)
		}
	}

	if len(systemParts) > 0 {
		args = append(args, "--system-prompt", strings.Join(systemParts, "\n\n"))
	}

	// Build prompt: if single user message, pass directly; if multi-turn, format conversation
	if len(convParts) == 1 && strings.HasPrefix(convParts[0], "Human: ") {
		prompt = strings.TrimPrefix(convParts[0], "Human: ")
	} else {
		prompt = strings.Join(convParts, "\n\n")
	}

	return args, prompt
}

// truncateErr returns the first 500 chars of an error message to prevent leaking large outputs.
func truncateErr(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 500 {
		return s[:500] + "..."
	}
	return s
}

// Complete sends a completion request via claude CLI.
func (p *ClaudeCLIProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()

	args, prompt := p.buildArgs(req, "json")

	cmd := exec.CommandContext(ctx, p.cliPath, args...)
	cmd.Stdin = strings.NewReader(prompt)
	cmd.Env = os.Environ()

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("claude cli: %w", ctx.Err())
		}
		errMsg := truncateErr(stderr.String())
		if errMsg == "" {
			errMsg = err.Error()
		}
		return nil, fmt.Errorf("claude cli: %s", errMsg)
	}

	var result cliResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("claude cli: parse output: %w", err)
	}

	if result.IsError {
		return nil, fmt.Errorf("claude cli: %s", result.Result)
	}

	model := p.model
	if req.Model != "" {
		model = req.Model
	}

	return &allm.Response{
		Provider:     "claude-cli",
		Model:        model,
		Content:      result.Result,
		InputTokens:  result.Usage.InputTokens + result.Usage.CacheReadInputTokens + result.Usage.CacheCreationInputTokens,
		OutputTokens: result.Usage.OutputTokens,
		Latency:      time.Since(start),
		FinishReason: "end_turn",
	}, nil
}

// Stream sends a streaming request via claude CLI (stream-json output).
func (p *ClaudeCLIProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		args, prompt := p.buildArgs(req, "stream-json")

		cmd := exec.CommandContext(ctx, p.cliPath, args...)
		cmd.Stdin = strings.NewReader(prompt)
		cmd.Env = os.Environ()

		var stderr bytes.Buffer
		cmd.Stderr = &stderr

		stdoutPipe, err := cmd.StdoutPipe()
		if err != nil {
			out <- allm.StreamChunk{Error: fmt.Errorf("claude cli: stdout pipe: %w", err)}
			return
		}

		if err := cmd.Start(); err != nil {
			out <- allm.StreamChunk{Error: fmt.Errorf("claude cli: start: %w", err)}
			return
		}

		scanner := bufio.NewScanner(stdoutPipe)
		// Increase buffer for large responses
		scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)

		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			// Quick type check before full parse
			var peek struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(line, &peek); err != nil {
				continue
			}

			switch peek.Type {
			case "assistant":
				var msg cliStreamMessage
				if err := json.Unmarshal(line, &msg); err != nil {
					continue
				}
				for _, block := range msg.Message.Content {
					if block.Type == "text" && block.Text != "" {
						out <- allm.StreamChunk{Content: block.Text}
					}
				}
			case "result":
				var result cliResult
				if err := json.Unmarshal(line, &result); err == nil && result.IsError {
					out <- allm.StreamChunk{Error: fmt.Errorf("claude cli: %s", result.Result)}
					return
				}
			}
		}

		if err := cmd.Wait(); err != nil {
			if ctx.Err() != nil {
				out <- allm.StreamChunk{Error: fmt.Errorf("claude cli: %w", ctx.Err())}
				return
			}
			errMsg := truncateErr(stderr.String())
			if errMsg == "" {
				errMsg = err.Error()
			}
			out <- allm.StreamChunk{Error: fmt.Errorf("claude cli: %s", errMsg)}
			return
		}

		out <- allm.StreamChunk{Done: true}
	}()

	return out
}

// Models returns a static list of available Claude models.
// The CLI doesn't provide a model listing endpoint.
func (p *ClaudeCLIProvider) Models(_ context.Context) ([]allm.Model, error) {
	models := []allm.Model{
		{ID: "claude-opus-4-20250514", Name: "Claude Opus 4", Provider: "claude-cli"},
		{ID: "claude-sonnet-4-20250514", Name: "Claude Sonnet 4", Provider: "claude-cli"},
		{ID: "claude-sonnet-4-5-20250929", Name: "Claude Sonnet 4.5", Provider: "claude-cli"},
		{ID: "claude-haiku-3-5-20241022", Name: "Claude Haiku 3.5", Provider: "claude-cli"},
	}
	return models, nil
}
