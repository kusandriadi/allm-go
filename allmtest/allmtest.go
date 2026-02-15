// Package allmtest provides test helpers for allm-go.
//
// Use MockProvider to test services that depend on allm.Provider
// without making real API calls:
//
//	mock := allmtest.NewMockProvider("test",
//	    allmtest.WithResponse(&allm.Response{Content: "Hello!"}),
//	)
//	client := allm.New(mock)
//	resp, err := client.Complete(ctx, "Hi")
//
// Capture and inspect requests:
//
//	mock := allmtest.NewMockProvider("test",
//	    allmtest.WithResponse(&allm.Response{Content: "OK"}),
//	)
//	client := allm.New(mock)
//	client.Complete(ctx, "Hello")
//
//	req := mock.LastRequest() // inspect what was sent
//	count := mock.CallCount() // how many times called
package allmtest

import (
	"context"
	"sync"

	"github.com/kusandriadi/allm-go"
)

// MockProvider implements allm.Provider, allm.ModelLister, and allm.Embedder for testing.
type MockProvider struct {
	mu            sync.Mutex
	name          string
	response      *allm.Response
	err           error
	chunks        []allm.StreamChunk
	models        []allm.Model
	embedResponse *allm.EmbedResponse
	requests      []*allm.Request
}

// MockOption configures the MockProvider.
type MockOption func(*MockProvider)

// WithResponse sets the response returned by Complete.
func WithResponse(resp *allm.Response) MockOption {
	return func(m *MockProvider) {
		m.response = resp
	}
}

// WithError sets the error returned by Complete.
func WithError(err error) MockOption {
	return func(m *MockProvider) {
		m.err = err
	}
}

// WithStreamChunks sets the chunks returned by Stream.
func WithStreamChunks(chunks []allm.StreamChunk) MockOption {
	return func(m *MockProvider) {
		m.chunks = chunks
	}
}

// WithModels sets the models returned by Models.
func WithModels(models []allm.Model) MockOption {
	return func(m *MockProvider) {
		m.models = models
	}
}

// WithEmbedResponse sets the response returned by Embed.
func WithEmbedResponse(resp *allm.EmbedResponse) MockOption {
	return func(m *MockProvider) {
		m.embedResponse = resp
	}
}

// NewMockProvider creates a new MockProvider with the given name and options.
func NewMockProvider(name string, opts ...MockOption) *MockProvider {
	m := &MockProvider{
		name: name,
		response: &allm.Response{
			Content:  "mock response",
			Provider: name,
			Model:    "mock-model",
		},
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Name returns the provider name.
func (m *MockProvider) Name() string {
	return m.name
}

// Available always returns true for mock provider.
func (m *MockProvider) Available() bool {
	return true
}

// Complete records the request and returns the configured response or error.
func (m *MockProvider) Complete(_ context.Context, req *allm.Request) (*allm.Response, error) {
	m.mu.Lock()
	m.requests = append(m.requests, req)
	m.mu.Unlock()

	if m.err != nil {
		return nil, m.err
	}
	return m.response, nil
}

// Stream records the request and returns configured chunks.
func (m *MockProvider) Stream(_ context.Context, req *allm.Request) <-chan allm.StreamChunk {
	m.mu.Lock()
	m.requests = append(m.requests, req)
	m.mu.Unlock()

	out := make(chan allm.StreamChunk)
	go func() {
		defer close(out)
		if m.err != nil {
			out <- allm.StreamChunk{Error: m.err}
			return
		}
		if len(m.chunks) > 0 {
			for _, chunk := range m.chunks {
				out <- chunk
			}
			return
		}
		// Default: send response as single chunk
		out <- allm.StreamChunk{Content: m.response.Content}
		out <- allm.StreamChunk{Done: true}
	}()
	return out
}

// Models returns configured models.
func (m *MockProvider) Models(_ context.Context) ([]allm.Model, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.models, nil
}

// Embed returns configured embeddings.
func (m *MockProvider) Embed(_ context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	m.mu.Lock()
	m.requests = append(m.requests, &allm.Request{Model: req.Model})
	m.mu.Unlock()

	if m.err != nil {
		return nil, m.err
	}
	if m.embedResponse != nil {
		return m.embedResponse, nil
	}
	// Default: return zero vectors matching input count
	embeddings := make([][]float64, len(req.Input))
	for i := range embeddings {
		embeddings[i] = make([]float64, 3)
	}
	return &allm.EmbedResponse{
		Embeddings: embeddings,
		Model:      "mock-embed",
		Provider:   m.name,
	}, nil
}

// LastRequest returns the most recent request, or nil if none.
func (m *MockProvider) LastRequest() *allm.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.requests) == 0 {
		return nil
	}
	return m.requests[len(m.requests)-1]
}

// Requests returns all recorded requests.
func (m *MockProvider) Requests() []*allm.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]*allm.Request, len(m.requests))
	copy(cp, m.requests)
	return cp
}

// CallCount returns the number of times Complete or Stream was called.
func (m *MockProvider) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests)
}

// Reset clears all recorded requests.
func (m *MockProvider) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = nil
}

// SetResponse updates the response for subsequent calls.
func (m *MockProvider) SetResponse(resp *allm.Response) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.response = resp
	m.err = nil
}

// SetError updates the error for subsequent calls.
func (m *MockProvider) SetError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.err = err
}
