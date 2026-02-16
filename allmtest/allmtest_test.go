package allmtest

import (
	"context"
	"errors"
	"testing"

	"github.com/kusandriadi/allm-go"
)

func TestMockProviderDefaults(t *testing.T) {
	m := NewMockProvider("test")

	if m.Name() != "test" {
		t.Errorf("expected 'test', got %q", m.Name())
	}
	if !m.Available() {
		t.Error("mock should always be available")
	}
	if m.CallCount() != 0 {
		t.Error("expected 0 calls")
	}
	if m.LastRequest() != nil {
		t.Error("expected nil last request")
	}
}

func TestMockProviderComplete(t *testing.T) {
	expected := &allm.Response{Content: "Hello!", Provider: "test"}
	m := NewMockProvider("test", WithResponse(expected))

	req := &allm.Request{
		Messages: []allm.Message{{Role: "user", Content: "Hi"}},
	}
	resp, err := m.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Content)
	}
	if m.CallCount() != 1 {
		t.Errorf("expected 1 call, got %d", m.CallCount())
	}
	if m.LastRequest() != req {
		t.Error("last request not captured")
	}
}

func TestMockProviderError(t *testing.T) {
	testErr := errors.New("api error")
	m := NewMockProvider("test", WithError(testErr))

	_, err := m.Complete(context.Background(), &allm.Request{
		Messages: []allm.Message{{Role: "user", Content: "Hi"}},
	})
	if err != testErr {
		t.Errorf("expected test error, got %v", err)
	}
}

func TestMockProviderStream(t *testing.T) {
	chunks := []allm.StreamChunk{
		{Content: "Hello"},
		{Content: " world"},
		{Done: true},
	}
	m := NewMockProvider("test", WithStreamChunks(chunks))

	var result string
	for chunk := range m.Stream(context.Background(), &allm.Request{
		Messages: []allm.Message{{Role: "user", Content: "Hi"}},
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

func TestMockProviderStreamDefault(t *testing.T) {
	m := NewMockProvider("test")

	var result string
	for chunk := range m.Stream(context.Background(), &allm.Request{
		Messages: []allm.Message{{Role: "user", Content: "Hi"}},
	}) {
		if chunk.Error != nil {
			t.Fatalf("unexpected error: %v", chunk.Error)
		}
		result += chunk.Content
	}
	if result != "mock response" {
		t.Errorf("expected 'mock response', got %q", result)
	}
}

func TestMockProviderModels(t *testing.T) {
	models := []allm.Model{
		{ID: "m1", Name: "Model 1", Provider: "test"},
	}
	m := NewMockProvider("test", WithModels(models))

	got, err := m.Models(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 model, got %d", len(got))
	}
	if got[0].ID != "m1" {
		t.Errorf("expected 'm1', got %q", got[0].ID)
	}
}

func TestMockProviderReset(t *testing.T) {
	m := NewMockProvider("test")

	m.Complete(context.Background(), &allm.Request{
		Messages: []allm.Message{{Role: "user", Content: "Hi"}},
	})
	if m.CallCount() != 1 {
		t.Fatal("expected 1 call before reset")
	}

	m.Reset()
	if m.CallCount() != 0 {
		t.Error("expected 0 calls after reset")
	}
	if m.LastRequest() != nil {
		t.Error("expected nil last request after reset")
	}
}

func TestMockProviderSetResponse(t *testing.T) {
	m := NewMockProvider("test")

	m.SetResponse(&allm.Response{Content: "updated"})
	resp, _ := m.Complete(context.Background(), &allm.Request{
		Messages: []allm.Message{{Role: "user", Content: "Hi"}},
	})
	if resp.Content != "updated" {
		t.Errorf("expected 'updated', got %q", resp.Content)
	}
}

func TestMockProviderSetError(t *testing.T) {
	m := NewMockProvider("test")

	testErr := errors.New("new error")
	m.SetError(testErr)
	_, err := m.Complete(context.Background(), &allm.Request{
		Messages: []allm.Message{{Role: "user", Content: "Hi"}},
	})
	if err != testErr {
		t.Errorf("expected test error, got %v", err)
	}
}

func TestMockProviderRequests(t *testing.T) {
	m := NewMockProvider("test")

	req1 := &allm.Request{Messages: []allm.Message{{Role: "user", Content: "Hi"}}}
	req2 := &allm.Request{Messages: []allm.Message{{Role: "user", Content: "Bye"}}}

	m.Complete(context.Background(), req1)
	m.Complete(context.Background(), req2)

	reqs := m.Requests()
	if len(reqs) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(reqs))
	}
	if reqs[0] != req1 {
		t.Error("first request not matched")
	}
	if reqs[1] != req2 {
		t.Error("second request not matched")
	}
}

func TestMockProviderWithAllmClient(t *testing.T) {
	m := NewMockProvider("test", WithResponse(&allm.Response{
		Content:  "Integration works!",
		Provider: "test",
		Model:    "mock-model",
	}))

	client := allm.New(m)
	resp, err := client.Complete(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Integration works!" {
		t.Errorf("expected 'Integration works!', got %q", resp.Content)
	}
}

func TestMockProviderConcurrency(t *testing.T) {
	m := NewMockProvider("test")

	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			m.Complete(context.Background(), &allm.Request{
				Messages: []allm.Message{{Role: "user", Content: "Hi"}},
			})
			done <- true
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
	if m.CallCount() != 10 {
		t.Errorf("expected 10 calls, got %d", m.CallCount())
	}
}

func TestMockProviderEmbed(t *testing.T) {
	m := NewMockProvider("test")

	resp, err := m.Embed(context.Background(), &allm.EmbedRequest{
		Input: []string{"hello", "world"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(resp.Embeddings))
	}
	// Default embed response returns zero vectors of length 3
	if len(resp.Embeddings[0]) != 3 {
		t.Errorf("expected vector length 3, got %d", len(resp.Embeddings[0]))
	}
	if resp.Provider != "test" {
		t.Errorf("expected provider 'test', got %q", resp.Provider)
	}
	if resp.Model != "mock-embed" {
		t.Errorf("expected model 'mock-embed', got %q", resp.Model)
	}
}

func TestMockProviderEmbedCustom(t *testing.T) {
	m := NewMockProvider("test", WithEmbedResponse(&allm.EmbedResponse{
		Embeddings: [][]float64{{0.1, 0.2, 0.3}},
		Model:      "custom-embed",
		Provider:   "test",
	}))

	resp, err := m.Embed(context.Background(), &allm.EmbedRequest{
		Input: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Embeddings))
	}
	if resp.Embeddings[0][0] != 0.1 {
		t.Errorf("expected 0.1, got %f", resp.Embeddings[0][0])
	}
	if resp.Model != "custom-embed" {
		t.Errorf("expected custom-embed, got %q", resp.Model)
	}
}

func TestMockProviderStreamError(t *testing.T) {
	testErr := errors.New("stream failure")
	m := NewMockProvider("test", WithError(testErr))

	var gotError error
	for chunk := range m.Stream(context.Background(), &allm.Request{
		Messages: []allm.Message{{Role: "user", Content: "Hi"}},
	}) {
		if chunk.Error != nil {
			gotError = chunk.Error
			break
		}
	}
	if gotError != testErr {
		t.Errorf("expected stream error, got %v", gotError)
	}
}
