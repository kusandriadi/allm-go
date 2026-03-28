package allm

import (
	"context"
	"errors"
	"testing"
)

// TestResponseFormatType tests ResponseFormat type validation
func TestResponseFormatType(t *testing.T) {
	tests := []struct {
		name    string
		format  *ResponseFormat
		wantErr bool
	}{
		{
			name:    "nil is valid",
			format:  nil,
			wantErr: false,
		},
		{
			name: "json_object is valid",
			format: &ResponseFormat{
				Type: ResponseFormatJSON,
			},
			wantErr: false,
		},
		{
			name: "json_schema with name and schema is valid",
			format: &ResponseFormat{
				Type:   ResponseFormatJSONSchema,
				Name:   "test_schema",
				Schema: map[string]any{"type": "object"},
			},
			wantErr: false,
		},
		{
			name: "json_schema without name is invalid",
			format: &ResponseFormat{
				Type:   ResponseFormatJSONSchema,
				Schema: map[string]any{"type": "object"},
			},
			wantErr: true,
		},
		{
			name: "json_schema without schema is invalid",
			format: &ResponseFormat{
				Type: ResponseFormatJSONSchema,
				Name: "test_schema",
			},
			wantErr: true,
		},
		{
			name: "invalid type",
			format: &ResponseFormat{
				Type: "invalid",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{
				Messages:       []Message{{Role: RoleUser, Content: "test"}},
				ResponseFormat: tt.format,
			}
			err := validateRequest(req)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestThinkingValidation tests ThinkingConfig validation
func TestThinkingValidation(t *testing.T) {
	tests := []struct {
		name     string
		thinking *ThinkingConfig
		wantErr  bool
	}{
		{
			name:     "nil is valid",
			thinking: nil,
			wantErr:  false,
		},
		{
			name: "positive budget is valid",
			thinking: &ThinkingConfig{
				Type:         "enabled",
				BudgetTokens: 1000,
			},
			wantErr: false,
		},
		{
			name: "zero budget is valid",
			thinking: &ThinkingConfig{
				Type:         "enabled",
				BudgetTokens: 0,
			},
			wantErr: false,
		},
		{
			name: "negative budget is invalid",
			thinking: &ThinkingConfig{
				Type:         "enabled",
				BudgetTokens: -100,
			},
			wantErr: true,
		},
		{
			name: "budget exceeding max is invalid",
			thinking: &ThinkingConfig{
				Type:         "enabled",
				BudgetTokens: MaxMaxTokens + 1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{
				Messages: []Message{{Role: RoleUser, Content: "test"}},
				Thinking: tt.thinking,
			}
			err := validateRequest(req)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestCacheControlField tests CacheControl field on Message
func TestCacheControlField(t *testing.T) {
	msg := Message{
		Role:    RoleUser,
		Content: "test",
		CacheControl: &CacheControl{
			Type: CacheEphemeral,
		},
	}

	if msg.CacheControl == nil {
		t.Error("CacheControl should not be nil")
	}
	if msg.CacheControl.Type != CacheEphemeral {
		t.Errorf("CacheControl.Type = %v, want %v", msg.CacheControl.Type, CacheEphemeral)
	}
}

// TestWithResponseFormat tests WithResponseFormat option
func TestWithResponseFormat(t *testing.T) {
	format := &ResponseFormat{
		Type: ResponseFormatJSON,
	}
	client := New(nil, WithResponseFormat(format))

	if client.responseFormat != format {
		t.Error("WithResponseFormat did not set responseFormat")
	}
}

// TestWithThinking tests WithThinking option
func TestWithThinking(t *testing.T) {
	client := New(nil, WithThinking(1000))

	if client.thinking == nil {
		t.Error("WithThinking did not set thinking")
	}
	if client.thinking.BudgetTokens != 1000 {
		t.Errorf("thinking.BudgetTokens = %d, want 1000", client.thinking.BudgetTokens)
	}
	if client.thinking.Type != "enabled" {
		t.Errorf("thinking.Type = %s, want enabled", client.thinking.Type)
	}
}

// TestWithMaxContextTokens tests WithMaxContextTokens option
func TestWithMaxContextTokens(t *testing.T) {
	client := New(nil, WithMaxContextTokens(5000))

	if client.maxContextTokens != 5000 {
		t.Errorf("maxContextTokens = %d, want 5000", client.maxContextTokens)
	}
}

// TestWithTruncationStrategy tests WithTruncationStrategy option
func TestWithTruncationStrategy(t *testing.T) {
	client := New(nil, WithTruncationStrategy(TruncateTail))

	if client.truncationStrategy != TruncateTail {
		t.Errorf("truncationStrategy = %s, want %s", client.truncationStrategy, TruncateTail)
	}
}

// TestCountTokensNotSupported tests CountTokens with unsupported provider
func TestCountTokensNotSupported(t *testing.T) {
	provider := &mockProvider{}
	client := New(provider)

	_, err := client.CountTokens(context.Background(), []Message{{Role: RoleUser, Content: "test"}})
	if err == nil {
		t.Error("CountTokens should return error for unsupported provider")
	}
	if !errors.Is(err, ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got: %v", err)
	}
}

// TestCreateBatchNotSupported tests CreateBatch with unsupported provider
func TestCreateBatchNotSupported(t *testing.T) {
	provider := &mockProvider{}
	client := New(provider)

	_, err := client.CreateBatch(context.Background(), []BatchRequest{{CustomID: "1", Messages: []Message{{Role: RoleUser, Content: "hi"}}}})
	if err == nil {
		t.Error("CreateBatch should return error for unsupported provider")
	}
	if !errors.Is(err, ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got: %v", err)
	}
}

// TestGetBatchNotSupported tests GetBatch with unsupported provider
func TestGetBatchNotSupported(t *testing.T) {
	provider := &mockProvider{}
	client := New(provider)

	_, err := client.GetBatch(context.Background(), "batch123")
	if err == nil {
		t.Error("GetBatch should return error for unsupported provider")
	}
	if !errors.Is(err, ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got: %v", err)
	}
}

// TestGenerateImageNotSupported tests GenerateImage with unsupported provider
func TestGenerateImageNotSupported(t *testing.T) {
	provider := &mockProvider{}
	client := New(provider)

	_, err := client.GenerateImage(context.Background(), "a cat")
	if err == nil {
		t.Error("GenerateImage should return error for unsupported provider")
	}
	if !errors.Is(err, ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got: %v", err)
	}
}

// TestGenerateImageEmptyPrompt tests GenerateImage with empty prompt
func TestGenerateImageEmptyPrompt(t *testing.T) {
	provider := &mockImageGenerator{}
	client := New(provider)

	_, err := client.GenerateImage(context.Background(), "")
	if !errors.Is(err, ErrEmptyInput) {
		t.Errorf("expected ErrEmptyInput, got %v", err)
	}
}

// TestGenerateImageValidation tests ImageRequest validation
func TestGenerateImageValidation(t *testing.T) {
	provider := &mockImageGenerator{}
	client := New(provider)

	// Excessive count
	_, err := client.GenerateImage(context.Background(), "test", WithImageCount(100))
	if err == nil {
		t.Error("expected error for excessive image count")
	}
}

// TestBatchValidation tests BatchRequest validation
func TestBatchValidation(t *testing.T) {
	tests := []struct {
		name    string
		reqs    []BatchRequest
		wantErr bool
	}{
		{"empty batch", []BatchRequest{}, true},
		{"empty custom_id", []BatchRequest{{Messages: []Message{{Role: RoleUser, Content: "hi"}}}}, true},
		{"duplicate custom_id", []BatchRequest{
			{CustomID: "a", Messages: []Message{{Role: RoleUser, Content: "hi"}}},
			{CustomID: "a", Messages: []Message{{Role: RoleUser, Content: "hello"}}},
		}, true},
		{"valid batch", []BatchRequest{
			{CustomID: "a", Messages: []Message{{Role: RoleUser, Content: "hi"}}},
		}, false}, // will fail with ErrNotSupported, but validation passes
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateBatchRequests(tt.reqs)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateBatchRequests() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestImageOptions tests image generation options
func TestImageOptions(t *testing.T) {
	req := &ImageRequest{Prompt: "test"}

	WithImageModel("dall-e-3")(req)
	if req.Model != "dall-e-3" {
		t.Errorf("Model = %s, want dall-e-3", req.Model)
	}

	WithImageSize(ImageSize1024)(req)
	if req.Size != ImageSize1024 {
		t.Errorf("Size = %s, want %s", req.Size, ImageSize1024)
	}

	WithImageQuality("hd")(req)
	if req.Quality != "hd" {
		t.Errorf("Quality = %s, want hd", req.Quality)
	}

	WithImageCount(3)(req)
	if req.N != 3 {
		t.Errorf("N = %d, want 3", req.N)
	}
}

// mockImageGenerator is a mock provider that supports image generation
type mockImageGenerator struct {
	mockProvider
}

func (m *mockImageGenerator) GenerateImage(ctx context.Context, req *ImageRequest) (*ImageResponse, error) {
	return &ImageResponse{
		Images: []GeneratedImage{
			{URL: "http://example.com/image.png"},
		},
		Provider: "mock",
		Model:    req.Model,
	}, nil
}

// TestTruncateMessagesWithoutTokenCounter tests truncation without token counting support
func TestTruncateMessagesWithoutTokenCounter(t *testing.T) {
	provider := &mockProvider{}
	s := clientState{
		provider:         provider,
		maxContextTokens: 100,
	}

	messages := []Message{
		{Role: RoleUser, Content: "message 1"},
		{Role: RoleUser, Content: "message 2"},
	}

	// Should return messages unchanged when provider doesn't support token counting
	truncated, err := truncateMessages(context.Background(), s, messages)
	if err != nil {
		t.Errorf("truncateMessages should not error when provider doesn't support counting: %v", err)
	}
	if len(truncated) != len(messages) {
		t.Errorf("truncateMessages should return unchanged messages when provider doesn't support counting")
	}
}

// TestResponseCacheTokens tests that Response has cache token fields
func TestResponseCacheTokens(t *testing.T) {
	resp := &Response{
		CacheReadTokens:  100,
		CacheWriteTokens: 50,
	}

	if resp.CacheReadTokens != 100 {
		t.Errorf("CacheReadTokens = %d, want 100", resp.CacheReadTokens)
	}
	if resp.CacheWriteTokens != 50 {
		t.Errorf("CacheWriteTokens = %d, want 50", resp.CacheWriteTokens)
	}
}

// TestResponseThinking tests that Response has thinking fields
func TestResponseThinking(t *testing.T) {
	resp := &Response{
		Thinking:       "Some thinking content",
		ThinkingTokens: 200,
	}

	if resp.Thinking != "Some thinking content" {
		t.Errorf("Thinking = %s, want 'Some thinking content'", resp.Thinking)
	}
	if resp.ThinkingTokens != 200 {
		t.Errorf("ThinkingTokens = %d, want 200", resp.ThinkingTokens)
	}
}

// TestSetResponseFormat tests runtime setter
func TestSetResponseFormat(t *testing.T) {
	client := New(nil)
	format := &ResponseFormat{Type: ResponseFormatJSON}
	client.SetResponseFormat(format)

	s := client.snapshot()
	if s.responseFormat != format {
		t.Error("SetResponseFormat did not set responseFormat")
	}

	client.SetResponseFormat(nil)
	s = client.snapshot()
	if s.responseFormat != nil {
		t.Error("SetResponseFormat(nil) should clear responseFormat")
	}
}

// TestSetThinking tests runtime setter
func TestSetThinking(t *testing.T) {
	client := New(nil)
	cfg := &ThinkingConfig{Type: "enabled", BudgetTokens: 5000}
	client.SetThinking(cfg)

	s := client.snapshot()
	if s.thinking != cfg {
		t.Error("SetThinking did not set thinking")
	}

	client.SetThinking(nil)
	s = client.snapshot()
	if s.thinking != nil {
		t.Error("SetThinking(nil) should clear thinking")
	}
}

// TestErrNotSupported tests sentinel error
func TestErrNotSupported(t *testing.T) {
	if !errors.Is(ErrNotSupported, ErrNotSupported) {
		t.Error("ErrNotSupported should match itself")
	}
}

// TestVersion tests that version was updated
func TestVersionUpdated(t *testing.T) {
	if Version != "0.8.12" {
		t.Errorf("Version = %s, want 0.8.12", Version)
	}
}
