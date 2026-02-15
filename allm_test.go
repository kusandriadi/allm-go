package allm

import (
	"bytes"
	"context"
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
