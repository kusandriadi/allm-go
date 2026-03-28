package provider

import (
	"testing"
)

// --- SSRF Prevention (A10) ---

func TestSSRF_BlocksLocalhostVariants(t *testing.T) {
	blocked := []string{
		"http://localhost/v1",
		"http://127.0.0.1/v1",
		"http://127.0.0.2/v1",
		"http://127.255.255.255/v1",
		"http://[::1]/v1",
		"http://0.0.0.0/v1",
		"http://[::]/v1",
	}
	for _, u := range blocked {
		if err := validateBaseURLProvider(u, false); err == nil {
			t.Errorf("expected SSRF block for %s", u)
		}
	}
}

func TestSSRF_BlocksPrivateIPs(t *testing.T) {
	blocked := []string{
		"http://10.0.0.1/v1",
		"http://10.255.255.255/v1",
		"http://172.16.0.1/v1",
		"http://172.31.255.255/v1",
		"http://192.168.0.1/v1",
		"http://192.168.255.255/v1",
	}
	for _, u := range blocked {
		if err := validateBaseURLProvider(u, false); err == nil {
			t.Errorf("expected SSRF block for %s", u)
		}
	}
}

func TestSSRF_BlocksLinkLocal(t *testing.T) {
	blocked := []string{
		"http://169.254.0.1/v1",
		"http://169.254.169.254/v1", // AWS metadata endpoint
		"http://[fe80::1]/v1",
	}
	for _, u := range blocked {
		if err := validateBaseURLProvider(u, false); err == nil {
			t.Errorf("expected SSRF block for %s", u)
		}
	}
}

func TestSSRF_BlocksIPv6MappedIPv4(t *testing.T) {
	blocked := []string{
		"http://[::ffff:127.0.0.1]/v1",
		"http://[::ffff:10.0.0.1]/v1",
		"http://[::ffff:192.168.1.1]/v1",
		"http://[::ffff:172.16.0.1]/v1",
	}
	for _, u := range blocked {
		if err := validateBaseURLProvider(u, false); err == nil {
			t.Errorf("expected SSRF block for IPv6-mapped IPv4: %s", u)
		}
	}
}

func TestSSRF_BlocksIPv6FullForm(t *testing.T) {
	blocked := []string{
		"http://[0000:0000:0000:0000:0000:0000:0000:0001]/v1", // ::1
		"http://[0:0:0:0:0:0:0:1]/v1",                         // ::1
		"http://[0:0:0:0:0:0:0:0]/v1",                         // ::
	}
	for _, u := range blocked {
		if err := validateBaseURLProvider(u, false); err == nil {
			t.Errorf("expected SSRF block for full-form IPv6: %s", u)
		}
	}
}

func TestSSRF_BlocksUserinfoInURL(t *testing.T) {
	blocked := []string{
		"http://evil@127.0.0.1/v1",
		"http://user:pass@10.0.0.1/v1",
		"https://admin@api.example.com/v1",
	}
	for _, u := range blocked {
		if err := validateBaseURLProvider(u, false); err == nil {
			t.Errorf("expected block for URL with userinfo: %s", u)
		}
	}
}

func TestSSRF_BlocksBadSchemes(t *testing.T) {
	blocked := []string{
		"file:///etc/passwd",
		"ftp://evil.com/data",
		"gopher://evil.com/",
		"javascript:alert(1)",
		"data:text/html,<h1>hi</h1>",
	}
	for _, u := range blocked {
		if err := validateBaseURLProvider(u, false); err == nil {
			t.Errorf("expected scheme block for %s", u)
		}
	}
}

func TestSSRF_AllowsPublicAPIs(t *testing.T) {
	allowed := []string{
		"https://api.openai.com/v1",
		"https://api.anthropic.com",
		"https://api.moonshot.cn/v1",
		"https://api.minimax.chat/v1",
		"https://api.z.ai/api/anthropic",
		"http://93.184.216.34/v1", // public IP
	}
	for _, u := range allowed {
		if err := validateBaseURLProvider(u, false); err != nil {
			t.Errorf("public URL should be allowed: %s (got: %v)", u, err)
		}
	}
}

func TestSSRF_AllowsLocalWhenPermitted(t *testing.T) {
	allowed := []string{
		"http://localhost:11434/v1",
		"http://127.0.0.1:8080/v1",
		"http://[::1]:8000/v1",
		"http://0.0.0.0:11434/v1",
	}
	for _, u := range allowed {
		if err := validateBaseURLProvider(u, true); err != nil {
			t.Errorf("local URL should be allowed when permitted: %s (got: %v)", u, err)
		}
	}
}

func TestSSRF_EmptyAndInvalid(t *testing.T) {
	if err := validateBaseURLProvider("", false); err == nil {
		t.Error("empty URL should be rejected")
	}
	if err := validateBaseURLProvider("not-a-url", false); err == nil {
		t.Error("non-URL should be rejected")
	}
	if err := validateBaseURLProvider("://missing-scheme", false); err == nil {
		t.Error("malformed URL should be rejected")
	}
}

// --- CLI Path Validation (A03) ---

func TestCLIPath_ValidPaths(t *testing.T) {
	valid := []string{
		"claude",
		"/usr/bin/claude",
		"/opt/claude-code/claude",
	}
	for _, p := range valid {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("valid cliPath %q should not panic: %v", p, r)
				}
			}()
			validateCLIPath(p)
		}()
	}
}

func TestCLIPath_RejectsPathTraversal(t *testing.T) {
	dangerous := []string{
		"../../bin/sh",
		"claude/../../../etc/passwd",
		"..\\cmd.exe",
	}
	for _, p := range dangerous {
		func() {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic for path traversal: %q", p)
				}
			}()
			validateCLIPath(p)
		}()
	}
}

func TestCLIPath_RejectsShellMetachars(t *testing.T) {
	dangerous := []string{
		"claude; rm -rf /",
		"claude && evil",
		"claude | cat /etc/passwd",
		"$(evil)",
		"`evil`",
		"claude\nrm -rf /",
		"claude with spaces",
		"claude'injection",
		`claude"injection`,
	}
	for _, p := range dangerous {
		func() {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic for shell metachar: %q", p)
				}
			}()
			validateCLIPath(p)
		}()
	}
}

func TestCLIPath_RejectsEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty cliPath")
		}
	}()
	validateCLIPath("")
}

// --- Provider SSRF integration ---

func TestAnthropicSSRF(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Anthropic with localhost baseURL should panic")
		}
	}()
	Anthropic("key", WithAnthropicBaseURL("http://localhost:8080"))
}

func TestOpenAISSRF(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("OpenAI with private IP baseURL should panic")
		}
	}()
	OpenAI("key", WithOpenAIBaseURL("http://10.0.0.1:8080"))
}

func TestCompatSSRF(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("compat provider with private IP should panic")
		}
	}()
	OpenAICompatible("test", "key", WithBaseURL("http://192.168.1.1:8080"))
}

func TestCompatSSRF_MetadataEndpoint(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("compat provider with AWS metadata endpoint should panic")
		}
	}()
	OpenAICompatible("test", "key", WithBaseURL("http://169.254.169.254/latest/meta-data/"))
}
