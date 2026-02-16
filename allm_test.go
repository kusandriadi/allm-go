package allm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

// mockProvider for testing
type mockProvider struct {
	mu        sync.Mutex
	name      string
	available bool
	response  *Response
	err       error
	chunks    []StreamChunk
	lastReq   *Request
}

func (m *mockProvider) Name() string {
	return m.name
}

func (m *mockProvider) Available() bool {
	return m.available
}

func (m *mockProvider) Complete(_ context.Context, req *Request) (*Response, error) {
	m.mu.Lock()
	m.lastReq = req
	err := m.err
	resp := m.response
	m.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (m *mockProvider) Stream(_ context.Context, req *Request) <-chan StreamChunk {
	m.mu.Lock()
	m.lastReq = req
	chunks := m.chunks
	m.mu.Unlock()
	out := make(chan StreamChunk)
	go func() {
		defer close(out)
		for _, chunk := range chunks {
			out <- chunk
		}
	}()
	return out
}

func (m *mockProvider) getLastReq() *Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastReq
}

// mockModelLister implements both Provider and ModelLister
type mockModelLister struct {
	mockProvider
	models []Model
}

func (m *mockModelLister) Models(_ context.Context) ([]Model, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.models, nil
}

// --- Client creation tests ---

func TestNew(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p)

	if c.provider != p {
		t.Error("provider not set")
	}
	if c.timeout != 60*time.Second {
		t.Errorf("expected 60s timeout, got %v", c.timeout)
	}
	if c.maxInputLen != 100000 {
		t.Errorf("expected 100000 maxInputLen, got %d", c.maxInputLen)
	}
}

func TestNewWithOptions(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p,
		WithTimeout(30*time.Second),
		WithMaxInputLen(50000),
		WithSystemPrompt("Be helpful."),
		WithModel("test-model"),
		WithMaxTokens(8192),
		WithTemperature(0.7),
	)

	if c.timeout != 30*time.Second {
		t.Errorf("expected 30s timeout, got %v", c.timeout)
	}
	if c.maxInputLen != 50000 {
		t.Errorf("expected 50000 maxInputLen, got %d", c.maxInputLen)
	}
	if c.systemPrompt != "Be helpful." {
		t.Errorf("expected system prompt, got %q", c.systemPrompt)
	}
	if c.model != "test-model" {
		t.Errorf("expected test-model, got %q", c.model)
	}
	if c.maxTokens != 8192 {
		t.Errorf("expected 8192, got %d", c.maxTokens)
	}
	if c.temperature != 0.7 {
		t.Errorf("expected 0.7, got %f", c.temperature)
	}
}

func TestNewNilProvider(t *testing.T) {
	c := New(nil)
	if c.provider != nil {
		t.Error("expected nil provider")
	}
}

// --- Complete tests ---

func TestCompleteNoProvider(t *testing.T) {
	c := &Client{}
	_, err := c.Complete(context.Background(), "test")
	if !errors.Is(err, ErrNoProvider) {
		t.Errorf("expected ErrNoProvider, got %v", err)
	}
}

func TestCompleteEmptyInput(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p)
	_, err := c.Complete(context.Background(), "")
	if !errors.Is(err, ErrEmptyInput) {
		t.Errorf("expected ErrEmptyInput, got %v", err)
	}
}

func TestCompleteInputTooLong(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p, WithMaxInputLen(10))

	longInput := "12345678901" // 11 chars > 10
	_, err := c.Complete(context.Background(), longInput)
	if !errors.Is(err, ErrInputTooLong) {
		t.Errorf("expected ErrInputTooLong, got %v", err)
	}
}

func TestCompleteInputExactLimit(t *testing.T) {
	expected := &Response{Content: "OK", Provider: "test"}
	p := &mockProvider{name: "test", available: true, response: expected}
	c := New(p, WithMaxInputLen(10))

	// Exactly 10 chars should work
	_, err := c.Complete(context.Background(), "1234567890")
	if err != nil {
		t.Errorf("expected no error at exact limit, got %v", err)
	}
}

func TestCompleteSuccess(t *testing.T) {
	expected := &Response{
		Content:      "Hello!",
		Provider:     "test",
		Model:        "test-model",
		InputTokens:  10,
		OutputTokens: 5,
	}
	p := &mockProvider{
		name:      "test",
		available: true,
		response:  expected,
	}
	c := New(p)

	resp, err := c.Complete(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != expected.Content {
		t.Errorf("expected %q, got %q", expected.Content, resp.Content)
	}
	if resp.Provider != "test" {
		t.Errorf("expected provider 'test', got %q", resp.Provider)
	}
}

func TestCompleteProviderError(t *testing.T) {
	provErr := errors.New("api error")
	p := &mockProvider{name: "test", available: true, err: provErr}
	c := New(p)

	_, err := c.Complete(context.Background(), "Hi")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err != provErr {
		t.Errorf("expected provider error, got %v", err)
	}
}

func TestCompleteContextCanceled(t *testing.T) {
	// Provider that respects context cancellation
	p := &mockProvider{
		name:      "test",
		available: true,
		err:       context.Canceled,
	}
	c := New(p)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := c.Complete(ctx, "Hi")
	if err == nil {
		t.Fatal("expected error for canceled context")
	}
	if !errors.Is(err, ErrCanceled) {
		t.Errorf("expected ErrCanceled, got %v", err)
	}
}

// --- Chat tests ---

func TestChatSuccess(t *testing.T) {
	expected := &Response{Content: "12", Provider: "test"}
	p := &mockProvider{name: "test", available: true, response: expected}
	c := New(p)

	resp, err := c.Chat(context.Background(), []Message{
		{Role: RoleUser, Content: "What is 2+2?"},
		{Role: RoleAssistant, Content: "4"},
		{Role: RoleUser, Content: "Multiply by 3"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "12" {
		t.Errorf("expected '12', got %q", resp.Content)
	}
}

func TestChatWithSystemPrompt(t *testing.T) {
	expected := &Response{Content: "OK"}
	p := &mockProvider{name: "test", available: true, response: expected}
	c := New(p, WithSystemPrompt("You are helpful."))

	_, err := c.Chat(context.Background(), []Message{
		{Role: RoleUser, Content: "Hello"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify system prompt was prepended
	if p.lastReq == nil {
		t.Fatal("expected request to be captured")
	}
	if len(p.lastReq.Messages) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(p.lastReq.Messages))
	}
	if p.lastReq.Messages[0].Role != RoleSystem {
		t.Errorf("expected first message to be system, got %q", p.lastReq.Messages[0].Role)
	}
	if p.lastReq.Messages[0].Content != "You are helpful." {
		t.Errorf("expected system prompt content, got %q", p.lastReq.Messages[0].Content)
	}
}

func TestChatEmptyMessages(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p)

	_, err := c.Chat(context.Background(), []Message{})
	if !errors.Is(err, ErrEmptyInput) {
		t.Errorf("expected ErrEmptyInput for empty messages, got %v", err)
	}
}

func TestChatNoProvider(t *testing.T) {
	c := &Client{}
	_, err := c.Chat(context.Background(), []Message{
		{Role: RoleUser, Content: "Hi"},
	})
	if !errors.Is(err, ErrNoProvider) {
		t.Errorf("expected ErrNoProvider, got %v", err)
	}
}

func TestChatInputTooLongWithImages(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p, WithMaxInputLen(10))

	_, err := c.Chat(context.Background(), []Message{
		{
			Role:    RoleUser,
			Content: "Hi",
			Images:  []Image{{MimeType: "image/png", Data: make([]byte, 20)}},
		},
	})
	if !errors.Is(err, ErrInputTooLong) {
		t.Errorf("expected ErrInputTooLong for large images, got %v", err)
	}
}

// --- Stream tests ---

func TestStream(t *testing.T) {
	p := &mockProvider{
		name:      "test",
		available: true,
		chunks: []StreamChunk{
			{Content: "Hello"},
			{Content: " world"},
			{Done: true},
		},
	}
	c := New(p)

	var result string
	for chunk := range c.Stream(context.Background(), []Message{
		{Role: RoleUser, Content: "Hi"},
	}) {
		if chunk.Error != nil {
			t.Fatalf("unexpected error: %v", chunk.Error)
		}
		result += chunk.Content
	}

	if result != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", result)
	}
}

func TestStreamError(t *testing.T) {
	testErr := errors.New("stream error")
	p := &mockProvider{
		name:      "test",
		available: true,
		chunks: []StreamChunk{
			{Content: "Hello"},
			{Error: testErr},
		},
	}
	c := New(p)

	var gotError error
	for chunk := range c.Stream(context.Background(), []Message{
		{Role: RoleUser, Content: "Hi"},
	}) {
		if chunk.Error != nil {
			gotError = chunk.Error
			break
		}
	}

	if gotError == nil {
		t.Error("expected error, got nil")
	}
}

func TestStreamNoProvider(t *testing.T) {
	c := &Client{}
	var gotError error
	for chunk := range c.Stream(context.Background(), []Message{
		{Role: RoleUser, Content: "Hi"},
	}) {
		if chunk.Error != nil {
			gotError = chunk.Error
			break
		}
	}
	if !errors.Is(gotError, ErrNoProvider) {
		t.Errorf("expected ErrNoProvider, got %v", gotError)
	}
}

func TestStreamEmptyInput(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p)

	var gotError error
	for chunk := range c.Stream(context.Background(), []Message{}) {
		if chunk.Error != nil {
			gotError = chunk.Error
			break
		}
	}
	if !errors.Is(gotError, ErrEmptyInput) {
		t.Errorf("expected ErrEmptyInput, got %v", gotError)
	}
}

func TestStreamInputTooLong(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p, WithMaxInputLen(5))

	var gotError error
	for chunk := range c.Stream(context.Background(), []Message{
		{Role: RoleUser, Content: "123456"},
	}) {
		if chunk.Error != nil {
			gotError = chunk.Error
			break
		}
	}
	if !errors.Is(gotError, ErrInputTooLong) {
		t.Errorf("expected ErrInputTooLong, got %v", gotError)
	}
}

// --- StreamToWriter tests ---

func TestStreamToWriter(t *testing.T) {
	p := &mockProvider{
		name:      "test",
		available: true,
		chunks: []StreamChunk{
			{Content: "Hello"},
			{Content: " world"},
			{Done: true},
		},
	}
	c := New(p)

	var buf bytes.Buffer
	err := c.StreamToWriter(context.Background(), []Message{
		{Role: RoleUser, Content: "Hi"},
	}, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if buf.String() != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", buf.String())
	}
}

func TestStreamToWriterError(t *testing.T) {
	testErr := errors.New("stream error")
	p := &mockProvider{
		name:      "test",
		available: true,
		chunks: []StreamChunk{
			{Error: testErr},
		},
	}
	c := New(p)

	var buf bytes.Buffer
	err := c.StreamToWriter(context.Background(), []Message{
		{Role: RoleUser, Content: "Hi"},
	}, &buf)
	if err == nil {
		t.Error("expected error, got nil")
	}
}

// --- Provider management tests ---

func TestSetProvider(t *testing.T) {
	p1 := &mockProvider{name: "p1", available: true}
	p2 := &mockProvider{name: "p2", available: true}
	c := New(p1)

	if c.Provider().Name() != "p1" {
		t.Error("initial provider wrong")
	}

	c.SetProvider(p2)
	if c.Provider().Name() != "p2" {
		t.Error("provider not updated")
	}
}

func TestSetSystemPrompt(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p)

	c.SetSystemPrompt("New prompt")
	if c.systemPrompt != "New prompt" {
		t.Error("system prompt not updated")
	}
}

// --- Models tests ---

func TestModelsSupported(t *testing.T) {
	p := &mockModelLister{
		mockProvider: mockProvider{name: "test", available: true},
		models: []Model{
			{ID: "model-1", Name: "Model One", Provider: "test"},
			{ID: "model-2", Name: "Model Two", Provider: "test"},
		},
	}
	c := New(p)

	models, err := c.Models(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].ID != "model-1" {
		t.Errorf("expected model-1, got %q", models[0].ID)
	}
}

func TestModelsNotSupported(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p)

	_, err := c.Models(context.Background())
	if err == nil {
		t.Error("expected error for provider without ModelLister")
	}
	if !strings.Contains(err.Error(), "does not support model listing") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestModelsNoProvider(t *testing.T) {
	c := &Client{}
	_, err := c.Models(context.Background())
	if !errors.Is(err, ErrNoProvider) {
		t.Errorf("expected ErrNoProvider, got %v", err)
	}
}

// --- Client-level model/maxTokens/temperature tests ---

func TestClientModelFlowsToRequest(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p, WithModel("my-model"), WithMaxTokens(2048), WithTemperature(0.5))

	c.Complete(context.Background(), "Hi")

	if p.lastReq == nil {
		t.Fatal("expected request to be captured")
	}
	if p.lastReq.Model != "my-model" {
		t.Errorf("expected model 'my-model', got %q", p.lastReq.Model)
	}
	if p.lastReq.MaxTokens != 2048 {
		t.Errorf("expected maxTokens 2048, got %d", p.lastReq.MaxTokens)
	}
	if p.lastReq.Temperature != 0.5 {
		t.Errorf("expected temperature 0.5, got %f", p.lastReq.Temperature)
	}
}

func TestClientModelDefaultsToZero(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p) // no model/maxTokens/temperature set

	c.Complete(context.Background(), "Hi")

	if p.lastReq.Model != "" {
		t.Errorf("expected empty model, got %q", p.lastReq.Model)
	}
	if p.lastReq.MaxTokens != 0 {
		t.Errorf("expected 0 maxTokens, got %d", p.lastReq.MaxTokens)
	}
	if p.lastReq.Temperature != 0 {
		t.Errorf("expected 0 temperature, got %f", p.lastReq.Temperature)
	}
}

func TestSetModel(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p, WithModel("model-a"))

	c.Complete(context.Background(), "Hi")
	if p.lastReq.Model != "model-a" {
		t.Errorf("expected model-a, got %q", p.lastReq.Model)
	}

	c.SetModel("model-b")
	c.Complete(context.Background(), "Hi")
	if p.lastReq.Model != "model-b" {
		t.Errorf("expected model-b, got %q", p.lastReq.Model)
	}
}

func TestStreamModelFlowsToRequest(t *testing.T) {
	p := &mockProvider{
		name: "test", available: true,
		chunks: []StreamChunk{{Content: "OK"}, {Done: true}},
	}
	c := New(p, WithModel("stream-model"), WithMaxTokens(1024))

	for range c.Stream(context.Background(), []Message{
		{Role: RoleUser, Content: "Hi"},
	}) {
	}

	if p.lastReq.Model != "stream-model" {
		t.Errorf("expected stream-model, got %q", p.lastReq.Model)
	}
	if p.lastReq.MaxTokens != 1024 {
		t.Errorf("expected 1024, got %d", p.lastReq.MaxTokens)
	}
}

// --- Message/Image struct tests ---

func TestMessageHasImages(t *testing.T) {
	m1 := Message{Role: RoleUser, Content: "Hi"}
	if len(m1.Images) != 0 {
		t.Error("expected no images")
	}

	m2 := Message{
		Role:    RoleUser,
		Content: "Look at this",
		Images:  []Image{{MimeType: "image/jpeg", Data: []byte{1, 2, 3}}},
	}
	if len(m2.Images) != 1 {
		t.Error("expected 1 image")
	}
}

func TestRoleConstants(t *testing.T) {
	if RoleSystem != "system" {
		t.Errorf("RoleSystem = %q", RoleSystem)
	}
	if RoleUser != "user" {
		t.Errorf("RoleUser = %q", RoleUser)
	}
	if RoleAssistant != "assistant" {
		t.Errorf("RoleAssistant = %q", RoleAssistant)
	}
}

func TestVersion(t *testing.T) {
	if Version == "" {
		t.Error("Version should not be empty")
	}
}

// --- Error sentinel tests ---

func TestStreamInputTooLongWithImages(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p, WithMaxInputLen(10))

	var gotError error
	for chunk := range c.Stream(context.Background(), []Message{
		{
			Role:    RoleUser,
			Content: "Hi",
			Images:  []Image{{MimeType: "image/png", Data: make([]byte, 20)}},
		},
	}) {
		if chunk.Error != nil {
			gotError = chunk.Error
			break
		}
	}
	if !errors.Is(gotError, ErrInputTooLong) {
		t.Errorf("expected ErrInputTooLong for large images in stream, got %v", gotError)
	}
}

// --- Concurrency tests ---

func TestConcurrentSetModelAndComplete(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p, WithModel("initial"))

	var wg sync.WaitGroup
	// Run concurrent SetModel calls
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			if n%2 == 0 {
				c.SetModel("model-a")
			} else {
				c.SetModel("model-b")
			}
		}(i)
	}
	// Run concurrent Complete calls
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.Complete(context.Background(), "Hi")
		}()
	}
	wg.Wait()
}

func TestConcurrentSetProviderAndChat(t *testing.T) {
	p1 := &mockProvider{name: "p1", available: true, response: &Response{Content: "OK"}}
	p2 := &mockProvider{name: "p2", available: true, response: &Response{Content: "OK"}}
	c := New(p1)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			c.SetProvider(p1)
			c.SetProvider(p2)
		}()
		go func() {
			defer wg.Done()
			c.Chat(context.Background(), []Message{
				{Role: RoleUser, Content: "Hello"},
			})
		}()
	}
	wg.Wait()
}

func TestConcurrentStream(t *testing.T) {
	p := &mockProvider{
		name: "test", available: true,
		chunks: []StreamChunk{{Content: "OK"}, {Done: true}},
	}
	c := New(p, WithModel("test-model"))

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			c.SetModel("changed")
			c.SetSystemPrompt("new prompt")
		}()
		go func() {
			defer wg.Done()
			for range c.Stream(context.Background(), []Message{
				{Role: RoleUser, Content: "Hi"},
			}) {
			}
		}()
	}
	wg.Wait()
}

// --- Embed tests ---

// mockEmbedder implements both Provider and Embedder
type mockEmbedder struct {
	mockProvider
	embedResp *EmbedResponse
}

func (m *mockEmbedder) Embed(_ context.Context, req *EmbedRequest) (*EmbedResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.err != nil {
		return nil, m.err
	}
	return m.embedResp, nil
}

func TestEmbedSupported(t *testing.T) {
	p := &mockEmbedder{
		mockProvider: mockProvider{name: "test", available: true},
		embedResp: &EmbedResponse{
			Embeddings: [][]float64{{0.1, 0.2, 0.3}},
			Model:      "embed-model",
			Provider:   "test",
		},
	}
	c := New(p)

	resp, err := c.Embed(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Embeddings))
	}
	if resp.Embeddings[0][0] != 0.1 {
		t.Errorf("expected 0.1, got %f", resp.Embeddings[0][0])
	}
}

func TestEmbedMultipleInputs(t *testing.T) {
	p := &mockEmbedder{
		mockProvider: mockProvider{name: "test", available: true},
		embedResp: &EmbedResponse{
			Embeddings: [][]float64{{0.1}, {0.2}},
			Model:      "embed-model",
			Provider:   "test",
		},
	}
	c := New(p)

	resp, err := c.Embed(context.Background(), "Hello", "World")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(resp.Embeddings))
	}
}

func TestEmbedNotSupported(t *testing.T) {
	p := &mockProvider{name: "test", available: true}
	c := New(p)

	_, err := c.Embed(context.Background(), "Hello")
	if err == nil {
		t.Fatal("expected error for provider without Embedder")
	}
	if !strings.Contains(err.Error(), "does not support embeddings") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestEmbedNoProvider(t *testing.T) {
	c := &Client{}
	_, err := c.Embed(context.Background(), "Hello")
	if !errors.Is(err, ErrNoProvider) {
		t.Errorf("expected ErrNoProvider, got %v", err)
	}
}

func TestEmbedEmptyInput(t *testing.T) {
	p := &mockEmbedder{
		mockProvider: mockProvider{name: "test", available: true},
	}
	c := New(p)

	_, err := c.Embed(context.Background())
	if !errors.Is(err, ErrEmptyInput) {
		t.Errorf("expected ErrEmptyInput, got %v", err)
	}
}

func TestEmbedModelFlowsToRequest(t *testing.T) {
	p := &mockEmbedder{
		mockProvider: mockProvider{name: "test", available: true},
		embedResp: &EmbedResponse{
			Embeddings: [][]float64{{0.1}},
			Model:      "custom-embed",
			Provider:   "test",
		},
	}
	c := New(p, WithEmbeddingModel("custom-embed"))

	resp, err := c.Embed(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Model != "custom-embed" {
		t.Errorf("expected custom-embed model, got %q", resp.Model)
	}
}

// --- Penalty params tests ---

func TestPenaltyParamsFlowToRequest(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p, WithPresencePenalty(0.5), WithFrequencyPenalty(0.3))

	c.Complete(context.Background(), "Hi")

	req := p.getLastReq()
	if req == nil {
		t.Fatal("expected request to be captured")
	}
	if req.PresencePenalty != 0.5 {
		t.Errorf("expected presence_penalty 0.5, got %f", req.PresencePenalty)
	}
	if req.FrequencyPenalty != 0.3 {
		t.Errorf("expected frequency_penalty 0.3, got %f", req.FrequencyPenalty)
	}
}

func TestPenaltyParamsInStream(t *testing.T) {
	p := &mockProvider{
		name: "test", available: true,
		chunks: []StreamChunk{{Content: "OK"}, {Done: true}},
	}
	c := New(p, WithPresencePenalty(0.8), WithFrequencyPenalty(0.2))

	for range c.Stream(context.Background(), []Message{
		{Role: RoleUser, Content: "Hi"},
	}) {
	}

	req := p.getLastReq()
	if req == nil {
		t.Fatal("expected request to be captured")
	}
	if req.PresencePenalty != 0.8 {
		t.Errorf("expected presence_penalty 0.8, got %f", req.PresencePenalty)
	}
	if req.FrequencyPenalty != 0.2 {
		t.Errorf("expected frequency_penalty 0.2, got %f", req.FrequencyPenalty)
	}
}

// --- Error sentinel tests ---

func TestErrorSentinels(t *testing.T) {
	errs := []struct {
		name string
		err  error
	}{
		{"ErrNoProvider", ErrNoProvider},
		{"ErrEmptyInput", ErrEmptyInput},
		{"ErrInputTooLong", ErrInputTooLong},
		{"ErrRateLimited", ErrRateLimited},
		{"ErrTimeout", ErrTimeout},
		{"ErrCanceled", ErrCanceled},
		{"ErrProvider", ErrProvider},
		{"ErrEmptyResponse", ErrEmptyResponse},
	}

	for _, tc := range errs {
		t.Run(tc.name, func(t *testing.T) {
			if tc.err == nil {
				t.Errorf("%s should not be nil", tc.name)
			}
			if tc.err.Error() == "" {
				t.Errorf("%s should have a message", tc.name)
			}
		})
	}
}

// --- Rate limit and empty response tests ---

func TestRateLimitedErrorPassesThrough(t *testing.T) {
	// Simulate a provider returning a rate limit wrapped error
	rateLimitErr := errors.Join(ErrRateLimited, errors.New("429 too many requests"))
	p := &mockProvider{name: "test", available: true, err: rateLimitErr}
	c := New(p)

	_, err := c.Complete(context.Background(), "Hello")
	if !errors.Is(err, ErrRateLimited) {
		t.Errorf("expected ErrRateLimited, got %v", err)
	}
}

func TestEmptyResponseError(t *testing.T) {
	// Simulate a provider returning an empty response error
	p := &mockProvider{name: "test", available: true, err: ErrEmptyResponse}
	c := New(p)

	_, err := c.Complete(context.Background(), "Hello")
	if !errors.Is(err, ErrEmptyResponse) {
		t.Errorf("expected ErrEmptyResponse, got %v", err)
	}
}

// --- Tool use tests ---

func TestToolTypes(t *testing.T) {
	tool := Tool{
		Name:        "get_weather",
		Description: "Get current weather",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []any{"city"},
		},
	}

	if tool.Name != "get_weather" {
		t.Error("tool name not set")
	}
	if tool.Parameters["type"] != "object" {
		t.Error("tool parameters not set")
	}
}

func TestToolCallInResponse(t *testing.T) {
	p := &mockProvider{
		name: "test", available: true,
		response: &Response{
			Content:      "",
			FinishReason: "tool_calls",
			ToolCalls: []ToolCall{
				{
					ID:        "call_123",
					Name:      "get_weather",
					Arguments: json.RawMessage(`{"city":"Jakarta"}`),
				},
			},
		},
	}
	c := New(p, WithTools(Tool{
		Name:        "get_weather",
		Description: "Get weather",
		Parameters:  map[string]any{"type": "object"},
	}))

	resp, err := c.Complete(context.Background(), "What's the weather in Jakarta?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "get_weather" {
		t.Errorf("expected get_weather, got %q", resp.ToolCalls[0].Name)
	}
	if resp.ToolCalls[0].ID != "call_123" {
		t.Errorf("expected call_123, got %q", resp.ToolCalls[0].ID)
	}
}

func TestToolsFlowToRequest(t *testing.T) {
	tools := []Tool{
		{Name: "fn1", Description: "First"},
		{Name: "fn2", Description: "Second"},
	}
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p, WithTools(tools...))

	c.Complete(context.Background(), "Hi")

	req := p.getLastReq()
	if req == nil {
		t.Fatal("expected request")
	}
	if len(req.Tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(req.Tools))
	}
	if req.Tools[0].Name != "fn1" {
		t.Errorf("expected fn1, got %q", req.Tools[0].Name)
	}
}

func TestSetTools(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p)

	c.SetTools(Tool{Name: "fn1", Description: "First"})
	c.Complete(context.Background(), "Hi")

	req := p.getLastReq()
	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(req.Tools))
	}
}

func TestToolResultMessageValidation(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p)

	// Message with only tool results (no text content) should pass validation
	_, err := c.Chat(context.Background(), []Message{
		{
			Role: RoleTool,
			ToolResults: []ToolResult{
				{ToolCallID: "call_123", Content: "32°C, sunny"},
			},
		},
	})
	if err != nil {
		t.Fatalf("tool result message should be valid, got: %v", err)
	}
}

func TestRoleTool(t *testing.T) {
	if RoleTool != "tool" {
		t.Errorf("expected 'tool', got %q", RoleTool)
	}
}

// --- Additional tool use tests ---

func TestChatWithToolMessages(t *testing.T) {
	// Simulate a full tool use conversation flow:
	// user → assistant with tool calls → tool results → assistant final answer
	p := &mockProvider{
		name: "test", available: true,
		response: &Response{Content: "The weather is 32°C and sunny."},
	}
	c := New(p)

	_, err := c.Chat(context.Background(), []Message{
		{Role: RoleUser, Content: "What's the weather in Jakarta?"},
		{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{
				{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Jakarta"}`)},
			},
		},
		{
			Role: RoleTool,
			ToolResults: []ToolResult{
				{ToolCallID: "call_1", Content: `{"temp":32,"condition":"sunny"}`},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	req := p.getLastReq()
	if req == nil {
		t.Fatal("expected request to be captured")
	}
	// System prompt not set, so messages should be: user + assistant + tool = 3
	// (with system prompt it would be 4)
	if len(req.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(req.Messages))
	}
	if req.Messages[2].Role != RoleTool {
		t.Errorf("expected tool role, got %q", req.Messages[2].Role)
	}
}

func TestValidateMessagesWithToolResults(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p, WithMaxInputLen(20))

	// Tool result content counts toward input length
	_, err := c.Chat(context.Background(), []Message{
		{
			Role: RoleTool,
			ToolResults: []ToolResult{
				{ToolCallID: "call_1", Content: strings.Repeat("x", 21)},
			},
		},
	})
	if !errors.Is(err, ErrInputTooLong) {
		t.Errorf("expected ErrInputTooLong for large tool result, got %v", err)
	}
}

func TestBuildRequestIncludesTools(t *testing.T) {
	tools := []Tool{
		{Name: "fn1", Description: "First"},
	}
	s := clientState{
		provider: &mockProvider{name: "test", available: true},
		tools:    tools,
	}

	msgs := []Message{{Role: RoleUser, Content: "Hi"}}
	req := buildRequest(msgs, s)

	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool in request, got %d", len(req.Tools))
	}
	if req.Tools[0].Name != "fn1" {
		t.Errorf("expected fn1, got %q", req.Tools[0].Name)
	}
}

func TestWithToolsOption(t *testing.T) {
	tools := []Tool{
		{Name: "fn1", Description: "First"},
		{Name: "fn2", Description: "Second"},
	}
	p := &mockProvider{name: "test", available: true}
	c := New(p, WithTools(tools...))

	if len(c.tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(c.tools))
	}
	if c.tools[0].Name != "fn1" {
		t.Errorf("expected fn1, got %q", c.tools[0].Name)
	}
	if c.tools[1].Name != "fn2" {
		t.Errorf("expected fn2, got %q", c.tools[1].Name)
	}
}

func TestConcurrentSetToolsAndComplete(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			c.SetTools(Tool{Name: "fn1", Description: "First"})
		}()
		go func() {
			defer wg.Done()
			c.Complete(context.Background(), "Hi")
		}()
	}
	wg.Wait()
}

func TestCompleteTimeout(t *testing.T) {
	slowProvider := &slowMockProvider{
		name:     "test",
		delay:    2 * time.Second,
		response: &Response{Content: "OK"},
	}
	c := New(slowProvider, WithTimeout(50*time.Millisecond))

	_, err := c.Complete(context.Background(), "Hi")
	if !errors.Is(err, ErrTimeout) {
		t.Errorf("expected ErrTimeout, got %v", err)
	}
}

// slowMockProvider simulates a provider that takes a long time to respond.
type slowMockProvider struct {
	name     string
	delay    time.Duration
	response *Response
}

func (m *slowMockProvider) Name() string    { return m.name }
func (m *slowMockProvider) Available() bool { return true }

func (m *slowMockProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	select {
	case <-time.After(m.delay):
		return m.response, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (m *slowMockProvider) Stream(_ context.Context, req *Request) <-chan StreamChunk {
	out := make(chan StreamChunk)
	close(out)
	return out
}

// --- Retry and Logging test helpers ---

// failNProvider fails the first N calls with the given error, then succeeds.
type failNProvider struct {
	mu        sync.Mutex
	name      string
	failCount int
	failErr   error
	response  *Response
	calls     int
}

func (p *failNProvider) Name() string    { return p.name }
func (p *failNProvider) Available() bool { return true }

func (p *failNProvider) Complete(_ context.Context, _ *Request) (*Response, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	if p.calls <= p.failCount {
		return nil, p.failErr
	}
	return p.response, nil
}

func (p *failNProvider) Stream(_ context.Context, _ *Request) <-chan StreamChunk {
	out := make(chan StreamChunk)
	close(out)
	return out
}

func (p *failNProvider) getCalls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

// failNEmbedder fails the first N Embed calls, then succeeds.
type failNEmbedder struct {
	failNProvider
	embedResp *EmbedResponse
	embedErr  error
	embedCall int
	embedFail int
}

func (e *failNEmbedder) Embed(_ context.Context, _ *EmbedRequest) (*EmbedResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.embedCall++
	if e.embedCall <= e.embedFail {
		return nil, e.embedErr
	}
	return e.embedResp, nil
}

// mockLogger records log calls.
type mockLogger struct {
	mu      sync.Mutex
	infos   []string
	warns   []string
	errors_ []string
}

func (l *mockLogger) Info(msg string, args ...any) {
	l.mu.Lock()
	l.infos = append(l.infos, msg)
	l.mu.Unlock()
}
func (l *mockLogger) Warn(msg string, args ...any) {
	l.mu.Lock()
	l.warns = append(l.warns, msg)
	l.mu.Unlock()
}
func (l *mockLogger) Error(msg string, args ...any) {
	l.mu.Lock()
	l.errors_ = append(l.errors_, msg)
	l.mu.Unlock()
}

func (l *mockLogger) infoCount() int  { l.mu.Lock(); defer l.mu.Unlock(); return len(l.infos) }
func (l *mockLogger) warnCount() int  { l.mu.Lock(); defer l.mu.Unlock(); return len(l.warns) }
func (l *mockLogger) errorCount() int { l.mu.Lock(); defer l.mu.Unlock(); return len(l.errors_) }

// --- Retry tests ---

func TestRetryOnRateLimit(t *testing.T) {
	p := &failNProvider{
		name:      "test",
		failCount: 1,
		failErr:   ErrRateLimited,
		response:  &Response{Content: "OK", Provider: "test"},
	}
	c := New(p,
		WithMaxRetries(2),
		WithRetryBaseDelay(1*time.Millisecond),
		WithRetryMaxDelay(10*time.Millisecond),
	)

	resp, err := c.Complete(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("expected success after retry, got %v", err)
	}
	if resp.Content != "OK" {
		t.Errorf("expected OK, got %q", resp.Content)
	}
	if p.getCalls() != 2 {
		t.Errorf("expected 2 calls, got %d", p.getCalls())
	}
}

func TestRetryOnTimeout(t *testing.T) {
	fp := &failNProvider{
		name:      "test",
		failCount: 1,
		failErr:   ErrTimeout,
		response:  &Response{Content: "OK", Provider: "test"},
	}

	c := New(fp,
		WithMaxRetries(2),
		WithRetryBaseDelay(1*time.Millisecond),
		WithRetryMaxDelay(10*time.Millisecond),
	)

	resp, err := c.Complete(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("expected success after retry, got %v", err)
	}
	if resp.Content != "OK" {
		t.Errorf("expected OK, got %q", resp.Content)
	}
	if fp.getCalls() != 2 {
		t.Errorf("expected 2 calls, got %d", fp.getCalls())
	}
}

func TestRetryExhausted(t *testing.T) {
	p := &failNProvider{
		name:      "test",
		failCount: 10, // always fail
		failErr:   ErrRateLimited,
		response:  &Response{Content: "OK"},
	}
	c := New(p,
		WithMaxRetries(2),
		WithRetryBaseDelay(1*time.Millisecond),
		WithRetryMaxDelay(10*time.Millisecond),
	)

	_, err := c.Complete(context.Background(), "Hi")
	if !errors.Is(err, ErrRateLimited) {
		t.Errorf("expected ErrRateLimited, got %v", err)
	}
	if p.getCalls() != 3 { // 1 initial + 2 retries
		t.Errorf("expected 3 calls, got %d", p.getCalls())
	}
}

func TestRetryNotOnCanceled(t *testing.T) {
	p := &failNProvider{
		name:      "test",
		failCount: 10,
		failErr:   ErrCanceled,
		response:  &Response{Content: "OK"},
	}
	c := New(p,
		WithMaxRetries(2),
		WithRetryBaseDelay(1*time.Millisecond),
	)

	_, err := c.Complete(context.Background(), "Hi")
	if !errors.Is(err, ErrCanceled) {
		t.Errorf("expected ErrCanceled, got %v", err)
	}
	// Should NOT retry — only 1 call
	if p.getCalls() != 1 {
		t.Errorf("expected 1 call (no retry for canceled), got %d", p.getCalls())
	}
}

func TestRetryNotOnEmptyInput(t *testing.T) {
	p := &mockProvider{name: "test", available: true, response: &Response{Content: "OK"}}
	c := New(p, WithMaxRetries(2))

	_, err := c.Complete(context.Background(), "")
	if !errors.Is(err, ErrEmptyInput) {
		t.Errorf("expected ErrEmptyInput, got %v", err)
	}
}

func TestRetryBackoffDelay(t *testing.T) {
	p := &failNProvider{
		name:      "test",
		failCount: 3,
		failErr:   ErrRateLimited,
		response:  &Response{Content: "OK"},
	}
	c := New(p,
		WithMaxRetries(3),
		WithRetryBaseDelay(10*time.Millisecond),
		WithRetryMaxDelay(1*time.Second),
	)

	start := time.Now()
	resp, err := c.Complete(context.Background(), "Hi")
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if resp.Content != "OK" {
		t.Errorf("expected OK, got %q", resp.Content)
	}
	// 3 retries: ~10ms + ~20ms + ~40ms = ~70ms minimum (without jitter max)
	// With 25% jitter max: ~87ms max
	// Be generous with bounds for CI
	if elapsed < 30*time.Millisecond {
		t.Errorf("expected backoff delay, but elapsed only %v", elapsed)
	}
}

func TestNoRetryByDefault(t *testing.T) {
	p := &failNProvider{
		name:      "test",
		failCount: 10,
		failErr:   ErrRateLimited,
		response:  &Response{Content: "OK"},
	}
	c := New(p) // no WithMaxRetries

	_, err := c.Complete(context.Background(), "Hi")
	if !errors.Is(err, ErrRateLimited) {
		t.Errorf("expected ErrRateLimited, got %v", err)
	}
	if p.getCalls() != 1 {
		t.Errorf("expected 1 call (no retries by default), got %d", p.getCalls())
	}
}

// --- Logger tests ---

func TestLoggerCalledOnSuccess(t *testing.T) {
	p := &mockProvider{
		name: "test", available: true,
		response: &Response{Content: "OK", Provider: "test"},
	}
	logger := &mockLogger{}
	c := New(p, WithLogger(logger))

	_, err := c.Complete(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if logger.infoCount() != 1 {
		t.Errorf("expected 1 info log, got %d", logger.infoCount())
	}
}

func TestLoggerCalledOnError(t *testing.T) {
	p := &mockProvider{
		name: "test", available: true,
		err: errors.New("provider error"),
	}
	logger := &mockLogger{}
	c := New(p, WithLogger(logger))

	_, err := c.Complete(context.Background(), "Hi")
	if err == nil {
		t.Fatal("expected error")
	}
	if logger.errorCount() != 1 {
		t.Errorf("expected 1 error log, got %d", logger.errorCount())
	}
}

// --- Hook tests ---

func TestHookCalledOnSuccess(t *testing.T) {
	p := &mockProvider{
		name: "test", available: true,
		response: &Response{Content: "OK", Provider: "test", InputTokens: 5, OutputTokens: 3},
	}
	var events []HookEvent
	var mu sync.Mutex
	hook := func(e HookEvent) {
		mu.Lock()
		events = append(events, e)
		mu.Unlock()
	}
	c := New(p, WithHook(hook))

	_, err := c.Complete(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 2 {
		t.Fatalf("expected 2 events (request + success), got %d", len(events))
	}
	if events[0].Type != HookRequest {
		t.Errorf("expected HookRequest, got %q", events[0].Type)
	}
	if events[1].Type != HookSuccess {
		t.Errorf("expected HookSuccess, got %q", events[1].Type)
	}
	if events[1].InputTokens != 5 {
		t.Errorf("expected 5 input tokens, got %d", events[1].InputTokens)
	}
}

func TestHookCalledOnRetry(t *testing.T) {
	p := &failNProvider{
		name:      "test",
		failCount: 1,
		failErr:   ErrRateLimited,
		response:  &Response{Content: "OK", Provider: "test"},
	}
	var events []HookEvent
	var mu sync.Mutex
	hook := func(e HookEvent) {
		mu.Lock()
		events = append(events, e)
		mu.Unlock()
	}
	c := New(p,
		WithMaxRetries(2),
		WithRetryBaseDelay(1*time.Millisecond),
		WithHook(hook),
	)

	_, err := c.Complete(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("expected success after retry, got %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	// Expected: HookRequest, HookRetry, HookSuccess
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d: %+v", len(events), events)
	}
	if events[0].Type != HookRequest {
		t.Errorf("expected HookRequest, got %q", events[0].Type)
	}
	if events[1].Type != HookRetry {
		t.Errorf("expected HookRetry, got %q", events[1].Type)
	}
	if events[1].Attempt != 2 {
		t.Errorf("expected attempt 2, got %d", events[1].Attempt)
	}
	if events[2].Type != HookSuccess {
		t.Errorf("expected HookSuccess, got %q", events[2].Type)
	}
}

// --- Embed retry test ---

func TestEmbedRetry(t *testing.T) {
	p := &failNEmbedder{
		failNProvider: failNProvider{name: "test"},
		embedFail:     1,
		embedErr:      ErrRateLimited,
		embedResp: &EmbedResponse{
			Embeddings: [][]float64{{0.1, 0.2}},
			Provider:   "test",
		},
	}
	c := New(p,
		WithMaxRetries(2),
		WithRetryBaseDelay(1*time.Millisecond),
	)

	resp, err := c.Embed(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("expected success after retry, got %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Embeddings))
	}
}
