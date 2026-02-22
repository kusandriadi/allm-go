package provider

import (
	"context"
	"os/exec"
	"strings"
	"testing"

	"github.com/kusandriadi/allm-go"
)

func TestClaudeCLINew(t *testing.T) {
	p := ClaudeCLI()
	if p.Name() != "claude-cli" {
		t.Errorf("expected 'claude-cli', got %q", p.Name())
	}
	if p.model != "claude-sonnet-4-20250514" {
		t.Errorf("expected default model, got %q", p.model)
	}
	if p.cliPath != "claude" {
		t.Errorf("expected default cli path 'claude', got %q", p.cliPath)
	}
}

func TestClaudeCLIWithOptions(t *testing.T) {
	p := ClaudeCLI(
		WithCLIModel("claude-opus-4-20250514"),
		WithCLIPath("/usr/local/bin/claude"),
		WithCLIEffort("high"),
	)
	if p.model != "claude-opus-4-20250514" {
		t.Error("model not set")
	}
	if p.cliPath != "/usr/local/bin/claude" {
		t.Error("cliPath not set")
	}
	if p.effort != "high" {
		t.Error("effort not set")
	}
}

func TestClaudeCLIAvailable(t *testing.T) {
	// Use a path that definitely doesn't exist
	p := ClaudeCLI(WithCLIPath("nonexistent-binary-xyz"))
	if p.Available() {
		t.Error("should not be available with nonexistent binary")
	}

	// Check if claude is actually installed
	if _, err := exec.LookPath("claude"); err == nil {
		p2 := ClaudeCLI()
		if !p2.Available() {
			t.Error("should be available when claude binary exists")
		}
	}
}

func TestClaudeCLIBuildArgs(t *testing.T) {
	p := ClaudeCLI(WithCLIModel("claude-opus-4"), WithCLIEffort("high"))

	req := &allm.Request{
		Messages: []allm.Message{
			{Role: allm.RoleSystem, Content: "You are helpful."},
			{Role: allm.RoleUser, Content: "Hello"},
		},
	}

	args, prompt := p.buildArgs(req, "json")

	// Check essential args
	hasFlag := func(flag string) bool {
		for _, a := range args {
			if a == flag {
				return true
			}
		}
		return false
	}

	if !hasFlag("-p") {
		t.Error("missing -p flag")
	}
	if !hasFlag("--no-session-persistence") {
		t.Error("missing --no-session-persistence flag")
	}
	if prompt != "Hello" {
		t.Errorf("expected prompt 'Hello', got %q", prompt)
	}

	// Check model is set
	foundModel := false
	for i, a := range args {
		if a == "--model" && i+1 < len(args) && args[i+1] == "claude-opus-4" {
			foundModel = true
		}
	}
	if !foundModel {
		t.Error("model not set in args")
	}

	// Check system prompt
	foundSystem := false
	for i, a := range args {
		if a == "--system-prompt" && i+1 < len(args) && args[i+1] == "You are helpful." {
			foundSystem = true
		}
	}
	if !foundSystem {
		t.Error("system prompt not set in args")
	}

	// Check effort
	foundEffort := false
	for i, a := range args {
		if a == "--effort" && i+1 < len(args) && args[i+1] == "high" {
			foundEffort = true
		}
	}
	if !foundEffort {
		t.Error("effort not set in args")
	}
}

func TestClaudeCLIBuildArgsMultiTurn(t *testing.T) {
	p := ClaudeCLI()

	req := &allm.Request{
		Messages: []allm.Message{
			{Role: allm.RoleUser, Content: "What is 2+2?"},
			{Role: allm.RoleAssistant, Content: "4"},
			{Role: allm.RoleUser, Content: "And 3+3?"},
		},
	}

	_, prompt := p.buildArgs(req, "json")

	expected := "Human: What is 2+2?\n\nAssistant: 4\n\nHuman: And 3+3?"
	if prompt != expected {
		t.Errorf("expected multi-turn prompt:\n%s\ngot:\n%s", expected, prompt)
	}
}

func TestClaudeCLIBuildArgsModelOverride(t *testing.T) {
	p := ClaudeCLI(WithCLIModel("claude-sonnet-4"))

	req := &allm.Request{
		Model: "claude-opus-4", // per-request override
		Messages: []allm.Message{
			{Role: allm.RoleUser, Content: "Hi"},
		},
	}

	args, _ := p.buildArgs(req, "json")

	for i, a := range args {
		if a == "--model" && i+1 < len(args) {
			if args[i+1] != "claude-opus-4" {
				t.Errorf("expected model override 'claude-opus-4', got %q", args[i+1])
			}
			return
		}
	}
	t.Error("--model flag not found")
}

func TestClaudeCLIBuildArgsStreamFormat(t *testing.T) {
	p := ClaudeCLI()

	req := &allm.Request{
		Messages: []allm.Message{
			{Role: allm.RoleUser, Content: "Hi"},
		},
	}

	args, _ := p.buildArgs(req, "stream-json")

	hasVerbose := false
	hasPartial := false
	for _, a := range args {
		if a == "--verbose" {
			hasVerbose = true
		}
		if a == "--include-partial-messages" {
			hasPartial = true
		}
	}
	if !hasVerbose {
		t.Error("stream-json format should include --verbose flag")
	}
	if !hasPartial {
		t.Error("stream-json format should include --include-partial-messages flag")
	}
}

func TestTruncateErr(t *testing.T) {
	short := "short error"
	if got := truncateErr(short); got != short {
		t.Errorf("expected %q, got %q", short, got)
	}

	long := strings.Repeat("x", 600)
	got := truncateErr(long)
	if len(got) != 503 { // 500 + "..."
		t.Errorf("expected truncated to 503 chars, got %d", len(got))
	}
	if !strings.HasSuffix(got, "...") {
		t.Error("expected ... suffix")
	}
}

func TestClaudeCLIModels(t *testing.T) {
	p := ClaudeCLI()
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) == 0 {
		t.Error("expected at least one model")
	}
	for _, m := range models {
		if m.Provider != "claude-cli" {
			t.Errorf("expected provider 'claude-cli', got %q", m.Provider)
		}
	}
}
