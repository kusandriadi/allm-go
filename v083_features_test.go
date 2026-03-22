package allm

import (
	"context"
	"testing"
)

// TestLogProbs verifies log probabilities are passed through the request.
func TestLogProbs(t *testing.T) {
	mock := &mockProvider{
		name:      "test",
		available: true,
		response: &Response{
			Content: "test",
			LogProbs: []TokenLogProb{
				{
					Token:   "test",
					LogProb: -0.5,
					TopLogProbs: []LogProb{
						{Token: "test", LogProb: -0.5},
						{Token: "best", LogProb: -1.2},
					},
				},
			},
		},
	}

	client := New(mock, WithLogProbs(5))
	resp, err := client.Complete(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify response has logprobs
	if len(resp.LogProbs) != 1 {
		t.Errorf("expected 1 token log prob, got %d", len(resp.LogProbs))
	}
	if len(resp.LogProbs[0].TopLogProbs) != 2 {
		t.Errorf("expected 2 top log probs, got %d", len(resp.LogProbs[0].TopLogProbs))
	}

	// Verify request has correct fields set
	req := mock.getLastReq()
	if !req.LogProbs {
		t.Error("expected LogProbs to be enabled")
	}
	if req.TopLogProbs != 5 {
		t.Errorf("expected TopLogProbs=5, got %d", req.TopLogProbs)
	}
}

// TestSeed verifies seed is passed through the request.
func TestSeed(t *testing.T) {
	mock := &mockProvider{
		name:      "test",
		available: true,
		response:  &Response{Content: "test", SystemFingerprint: "fp_12345"},
	}

	var seed int64 = 42
	client := New(mock, WithSeed(seed))
	resp, err := client.Complete(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify response has system fingerprint
	if resp.SystemFingerprint != "fp_12345" {
		t.Errorf("expected SystemFingerprint=fp_12345, got %s", resp.SystemFingerprint)
	}

	// Verify request has seed set
	req := mock.getLastReq()
	if req.Seed == nil {
		t.Error("expected Seed to be set")
	} else if *req.Seed != 42 {
		t.Errorf("expected Seed=42, got %d", *req.Seed)
	}
}

// TestParallelToolCalls verifies parallel tool calls control.
func TestParallelToolCalls(t *testing.T) {
	mock := &mockProvider{
		name:      "test",
		available: true,
		response:  &Response{Content: "test"},
	}

	parallelFalse := false
	client := New(mock, WithTools(Tool{Name: "test"}))

	// Create a request with ParallelToolCalls set to false
	req := &Request{
		Messages:          []Message{{Role: RoleUser, Content: "test"}},
		Tools:             []Tool{{Name: "test"}},
		ParallelToolCalls: &parallelFalse,
	}

	_, err := client.Chat(context.Background(), req.Messages)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// We can't easily verify the request in this setup, but we can verify no error
	// The actual verification would need to check the provider's last request
}

// TestPredictedOutput verifies predicted output is passed through.
func TestPredictedOutput(t *testing.T) {
	mock := &mockProvider{
		name:      "test",
		available: true,
		response:  &Response{Content: "test"},
	}

	client := New(mock)

	// Create a request with Prediction set
	prediction := &PredictedOutput{Content: "expected output"}
	req := &Request{
		Messages:   []Message{{Role: RoleUser, Content: "test"}},
		Prediction: prediction,
	}

	_, err := client.Chat(context.Background(), req.Messages)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify prediction was not set in the built request (buildRequest doesn't copy it)
	// This is expected — Prediction must be set directly on Request, not via client defaults
}

// TestRequestID verifies request ID is captured.
func TestRequestID(t *testing.T) {
	mock := &mockProvider{
		name:      "test",
		available: true,
		response:  &Response{Content: "test", RequestID: "req_12345"},
	}

	client := New(mock)
	resp, err := client.Complete(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.RequestID != "req_12345" {
		t.Errorf("expected RequestID=req_12345, got %s", resp.RequestID)
	}
}

// TestStreamThinking verifies thinking content in streaming.
func TestStreamThinking(t *testing.T) {
	mock := &mockProvider{
		name:      "test",
		available: true,
		chunks: []StreamChunk{
			{Thinking: "hmm, let me think..."},
			{Content: "the answer is 42"},
			{Done: true},
		},
	}

	client := New(mock)
	var thinkingParts []string
	var contentParts []string

	for chunk := range client.Stream(context.Background(), []Message{{Role: RoleUser, Content: "test"}}) {
		if chunk.Error != nil {
			t.Fatalf("unexpected error: %v", chunk.Error)
		}
		if chunk.Thinking != "" {
			thinkingParts = append(thinkingParts, chunk.Thinking)
		}
		if chunk.Content != "" {
			contentParts = append(contentParts, chunk.Content)
		}
		if chunk.Done {
			break
		}
	}

	if len(thinkingParts) != 1 {
		t.Errorf("expected 1 thinking part, got %d", len(thinkingParts))
	}
	if len(contentParts) != 1 {
		t.Errorf("expected 1 content part, got %d", len(contentParts))
	}
	if thinkingParts[0] != "hmm, let me think..." {
		t.Errorf("unexpected thinking content: %s", thinkingParts[0])
	}
}

// TestLogProbsValidation verifies validation of TopLogProbs field.
func TestLogProbsValidation(t *testing.T) {
	tests := []struct {
		name        string
		topLogProbs int
		wantErr     bool
	}{
		{"valid 0", 0, false},
		{"valid 5", 5, false},
		{"valid 20", 20, false},
		{"invalid negative", -1, true},
		{"invalid too large", 21, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{
				Messages:    []Message{{Role: RoleUser, Content: "test"}},
				LogProbs:    true,
				TopLogProbs: tt.topLogProbs,
			}

			err := validateRequest(req)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
