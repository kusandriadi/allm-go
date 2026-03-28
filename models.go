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
		fmt.Fprintf(&sb, "\n**%s** (%d models):\n", strings.ToUpper(providerName), len(modelList))
		for _, m := range modelList {
			fmt.Fprintf(&sb, "• `%s`", m.ID)
			if m.ContextWindow > 0 {
				if len(modelList) > 10 {
					fmt.Fprintf(&sb, " (%dk ctx)", m.ContextWindow/1000)
				} else {
					fmt.Fprintf(&sb, " - %dk context", m.ContextWindow/1000)
				}
			}
			if len(modelList) <= 10 && len(m.Capabilities) > 0 {
				fmt.Fprintf(&sb, " [%s]", strings.Join(m.Capabilities, ", "))
			}
			sb.WriteString("\n")
		}
	}

	return sb.String()
}
