package provider

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/kusandriadi/allm-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// OpenAIProvider implements allm.Provider for OpenAI GPT models.
type OpenAIProvider struct {
	apiKey      string
	model       string
	maxTokens   int
	temperature float64
	baseURL     string
	client      openai.Client
	logger      allm.Logger
}

// OpenAIOption configures the OpenAI provider.
type OpenAIOption func(*OpenAIProvider)

// WithOpenAIModel sets the model.
func WithOpenAIModel(model string) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.model = model
	}
}

// WithOpenAIMaxTokens sets max output tokens.
func WithOpenAIMaxTokens(n int) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.maxTokens = n
	}
}

// WithOpenAITemperature sets the temperature.
func WithOpenAITemperature(t float64) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.temperature = t
	}
}

// WithOpenAIBaseURL sets a custom base URL (for Azure, proxies).
func WithOpenAIBaseURL(url string) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.baseURL = url
	}
}

// WithOpenAILogger sets a logger for provider-level debug tracing.
func WithOpenAILogger(logger allm.Logger) OpenAIOption {
	return func(p *OpenAIProvider) {
		p.logger = logger
	}
}

// OpenAI creates a new OpenAI provider.
// If apiKey is empty, it reads from OPENAI_API_KEY environment variable.
func OpenAI(apiKey string, opts ...OpenAIOption) *OpenAIProvider {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	p := &OpenAIProvider{
		apiKey:    apiKey,
		model:     "gpt-4o",
		maxTokens: 4096,
	}

	for _, opt := range opts {
		opt(p)
	}

	p.client = p.buildClient()

	return p
}

// Name returns the provider name.
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// Available returns true if the API key is set.
func (p *OpenAIProvider) Available() bool {
	return p.apiKey != ""
}

// Complete sends a completion request.
func (p *OpenAIProvider) Complete(ctx context.Context, req *allm.Request) (*allm.Response, error) {
	start := time.Now()
	model := resolveModel(req.Model, p.model)

	if p.logger != nil {
		p.logger.Debug("provider complete",
			"provider", "openai",
			"model", model,
			"messages", len(req.Messages),
		)
	}

	messages, err := convertToOpenAI(req.Messages)
	if err != nil {
		return nil, err
	}
	params := openaiChatParams(messages, p.model, p.maxTokens, p.temperature, req)

	completion, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		if p.logger != nil {
			p.logger.Debug("provider complete failed",
				"provider", "openai",
				"model", model,
				"latency", time.Since(start),
				"error", sanitizeProviderError(err),
			)
		}
		return nil, wrapOpenAIError(err)
	}

	resp, respErr := openaiCompleteResponse(completion, "openai", model, start)
	if p.logger != nil && resp != nil {
		p.logger.Debug("provider complete done",
			"provider", "openai",
			"model", model,
			"latency", resp.Latency,
			"input_tokens", resp.InputTokens,
			"output_tokens", resp.OutputTokens,
			"finish_reason", resp.FinishReason,
		)
	}
	return resp, respErr
}

// Models returns available models from OpenAI.
func (p *OpenAIProvider) Models(ctx context.Context) ([]allm.Model, error) {
	return openaiListModels(ctx, p.client, "openai")
}

// Embed generates embeddings using the OpenAI Embeddings API.
func (p *OpenAIProvider) Embed(ctx context.Context, req *allm.EmbedRequest) (*allm.EmbedResponse, error) {
	return openaiEmbed(ctx, p.client, req, "text-embedding-3-small", "openai")
}

// Stream sends a real streaming request using the OpenAI SDK.
func (p *OpenAIProvider) Stream(ctx context.Context, req *allm.Request) <-chan allm.StreamChunk {
	out := make(chan allm.StreamChunk)

	go func() {
		defer close(out)

		if p.logger != nil {
			p.logger.Debug("provider stream",
				"provider", "openai",
				"model", resolveModel(req.Model, p.model),
				"messages", len(req.Messages),
			)
		}

		messages, err := convertToOpenAI(req.Messages)
		if err != nil {
			out <- allm.StreamChunk{Error: err}
			return
		}
		params := openaiChatParams(messages, p.model, p.maxTokens, p.temperature, req)

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)
		openaiStreamLoop(stream, out)
	}()

	return out
}

// GenerateImage creates images from a text prompt using DALL-E.
func (p *OpenAIProvider) GenerateImage(ctx context.Context, req *allm.ImageRequest) (*allm.ImageResponse, error) {
	start := time.Now()

	model := "dall-e-3"
	if req.Model != "" {
		model = req.Model
	}

	if p.logger != nil {
		p.logger.Debug("provider generate image",
			"provider", "openai",
			"model", model,
			"size", req.Size,
			"n", req.N,
		)
	}

	params := openai.ImageGenerateParams{
		Prompt: req.Prompt,
		Model:  openai.ImageModel(model),
	}

	if req.Size != "" {
		params.Size = openai.ImageGenerateParamsSize(req.Size)
	}
	if req.Quality != "" {
		params.Quality = openai.ImageGenerateParamsQuality(req.Quality)
	}
	if req.N > 0 {
		params.N = openai.Int(int64(req.N))
	}

	result, err := p.client.Images.Generate(ctx, params)
	if err != nil {
		if p.logger != nil {
			p.logger.Debug("provider generate image failed",
				"provider", "openai",
				"model", model,
				"error", sanitizeProviderError(err),
			)
		}
		return nil, wrapOpenAIError(err)
	}

	resp := &allm.ImageResponse{
		Provider: "openai",
		Model:    model,
		Latency:  time.Since(start),
	}

	for _, img := range result.Data {
		resp.Images = append(resp.Images, allm.GeneratedImage{
			URL:           img.URL,
			RevisedPrompt: img.RevisedPrompt,
		})
	}

	if p.logger != nil {
		p.logger.Debug("provider generate image done",
			"provider", "openai",
			"model", model,
			"latency", resp.Latency,
			"images", len(resp.Images),
		)
	}

	return resp, nil
}

// CreateBatch submits a batch of requests for processing.
// Note: This is a basic stub implementation. Full batch API requires file upload and polling.
func (p *OpenAIProvider) CreateBatch(ctx context.Context, requests []allm.BatchRequest) (*allm.Batch, error) {
	// TODO: Implement full batch API with file upload
	return nil, fmt.Errorf("openai: batch API not yet implemented")
}

// GetBatch retrieves the status and results of a batch job.
// Note: This is a basic stub implementation.
func (p *OpenAIProvider) GetBatch(ctx context.Context, batchID string) (*allm.Batch, error) {
	// TODO: Implement batch retrieval
	return nil, fmt.Errorf("openai: batch API not yet implemented")
}

// Speak converts text to speech using OpenAI TTS.
func (p *OpenAIProvider) Speak(ctx context.Context, req *allm.SpeechRequest) (*allm.SpeechResponse, error) {
	start := time.Now()

	model := "tts-1"
	if req.Model != "" {
		model = req.Model
	}

	voice := "alloy"
	if req.Voice != "" {
		voice = req.Voice
	}

	if p.logger != nil {
		p.logger.Debug("provider speak",
			"provider", "openai",
			"model", model,
			"voice", voice,
		)
	}

	params := openai.AudioSpeechNewParams{
		Model: openai.SpeechModel(model),
		Input: req.Input,
		Voice: openai.AudioSpeechNewParamsVoiceUnion{
			OfString: openai.String(voice),
		},
	}

	if req.Format != "" {
		params.ResponseFormat = openai.AudioSpeechNewParamsResponseFormat(req.Format)
	}
	if req.Speed > 0 {
		params.Speed = openai.Float(req.Speed)
	}

	httpResp, err := p.client.Audio.Speech.New(ctx, params)
	if err != nil {
		if p.logger != nil {
			p.logger.Debug("provider speak failed",
				"provider", "openai",
				"model", model,
				"error", sanitizeProviderError(err),
			)
		}
		return nil, wrapOpenAIError(err)
	}
	defer httpResp.Body.Close()

	// Read audio data
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, httpResp.Body); err != nil {
		return nil, fmt.Errorf("openai: failed to read audio data: %w", err)
	}

	format := req.Format
	if format == "" {
		format = "mp3" // OpenAI default
	}

	result := &allm.SpeechResponse{
		Audio:    buf.Bytes(),
		Format:   format,
		Provider: "openai",
		Model:    model,
		Latency:  time.Since(start),
	}

	if p.logger != nil {
		p.logger.Debug("provider speak done",
			"provider", "openai",
			"model", model,
			"latency", result.Latency,
			"bytes", len(result.Audio),
		)
	}

	return result, nil
}

// Transcribe converts speech to text using OpenAI Whisper.
func (p *OpenAIProvider) Transcribe(ctx context.Context, req *allm.TranscribeRequest) (*allm.TranscribeResponse, error) {
	start := time.Now()

	model := "whisper-1"
	if req.Model != "" {
		model = req.Model
	}

	if p.logger != nil {
		p.logger.Debug("provider transcribe",
			"provider", "openai",
			"model", model,
			"format", req.Format,
		)
	}

	// OpenAI SDK expects an io.Reader for the file
	params := openai.AudioTranscriptionNewParams{
		Model: openai.AudioModel(model),
		File:  bytes.NewReader(req.Audio),
	}

	if req.Language != "" {
		params.Language = openai.String(req.Language)
	}
	if req.Prompt != "" {
		params.Prompt = openai.String(req.Prompt)
	}

	transcription, err := p.client.Audio.Transcriptions.New(ctx, params)
	if err != nil {
		if p.logger != nil {
			p.logger.Debug("provider transcribe failed",
				"provider", "openai",
				"model", model,
				"error", sanitizeProviderError(err),
			)
		}
		return nil, wrapOpenAIError(err)
	}

	result := &allm.TranscribeResponse{
		Text:     transcription.Text,
		Language: req.Language, // OpenAI doesn't return detected language in basic response
		Provider: "openai",
		Model:    model,
		Latency:  time.Since(start),
	}

	if p.logger != nil {
		p.logger.Debug("provider transcribe done",
			"provider", "openai",
			"model", model,
			"latency", result.Latency,
		)
	}

	return result, nil
}

func (p *OpenAIProvider) buildClient() openai.Client {
	// Validate custom base URL for security (SSRF prevention)
	if p.baseURL != "" {
		if err := validateBaseURLProvider(p.baseURL, false); err != nil {
			panic(fmt.Sprintf("openai: %v", err))
		}
	}

	opts := []option.RequestOption{
		option.WithAPIKey(p.apiKey),
	}
	if p.baseURL != "" {
		opts = append(opts, option.WithBaseURL(p.baseURL))
	}
	return openai.NewClient(opts...)
}
