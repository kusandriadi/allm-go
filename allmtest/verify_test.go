package allmtest_test

import (
	"testing"

	"github.com/kusandriadi/allm-go"
	"github.com/kusandriadi/allm-go/allmtest"
)

func TestVerifyWithMock(t *testing.T) {
	mock := allmtest.NewMockProvider("test",
		allmtest.WithResponse(&allm.Response{
			Content:      "hello",
			Provider:     "test",
			Model:        "test-model",
			InputTokens:  10,
			OutputTokens: 5,
		}),
		allmtest.WithStreamChunks([]allm.StreamChunk{
			{Content: "hel"},
			{Content: "lo"},
			{Done: true},
		}),
	)
	client := allm.New(mock)

	allmtest.Verify(t, client,
		allmtest.SkipVision(),
		allmtest.SkipEmbeddings(),
		allmtest.SkipToolUse(),
		allmtest.SkipModels(),
	)
}

func TestVerifyOptions(t *testing.T) {
	// Just verify options don't panic
	opts := []allmtest.VerifyOption{
		allmtest.SkipVision(),
		allmtest.SkipEmbeddings(),
		allmtest.SkipToolUse(),
		allmtest.SkipStreaming(),
		allmtest.SkipModels(),
	}
	if len(opts) != 5 {
		t.Fatal("expected 5 options")
	}
}

func TestTruncate(t *testing.T) {
	// Verify truncate doesn't crash (it's unexported, tested via Verify output)
	mock := allmtest.NewMockProvider("test",
		allmtest.WithResponse(&allm.Response{
			Content:  "this is a very long response that should get truncated in the log output for readability",
			Provider: "test",
		}),
	)
	client := allm.New(mock)
	allmtest.Verify(t, client,
		allmtest.SkipVision(),
		allmtest.SkipEmbeddings(),
		allmtest.SkipToolUse(),
		allmtest.SkipModels(),
		allmtest.SkipStreaming(),
	)
}
