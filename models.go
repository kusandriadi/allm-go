package allm

import (
	"fmt"
	"sort"
	"strings"
)

// FormatModelList formats a provider→models map for display.
// Output uses markdown-style formatting suitable for chat platforms.
func FormatModelList(models map[string][]Model) string {
	var sb strings.Builder

	providers := make([]string, 0, len(models))
	for p := range models {
		providers = append(providers, p)
	}
	sort.Strings(providers)

	for _, providerName := range providers {
		modelList := models[providerName]
		sb.WriteString(fmt.Sprintf("\n**%s** (%d models):\n", strings.ToUpper(providerName), len(modelList)))
		for _, m := range modelList {
			sb.WriteString(fmt.Sprintf("• `%s`", m.ID))
			if m.ContextWindow > 0 {
				if len(modelList) > 10 {
					sb.WriteString(fmt.Sprintf(" (%dk ctx)", m.ContextWindow/1000))
				} else {
					sb.WriteString(fmt.Sprintf(" - %dk context", m.ContextWindow/1000))
				}
			}
			if len(modelList) <= 10 && len(m.Capabilities) > 0 {
				sb.WriteString(fmt.Sprintf(" [%s]", strings.Join(m.Capabilities, ", ")))
			}
			sb.WriteString("\n")
		}
	}

	return sb.String()
}
