// backend-select lists the compute devices, requests an explicit backend and
// falls back cleanly when it is not there.
//
//	go run ./examples/backend-select [model.gguf]
//
// Device discovery needs no model, so the first half always runs.
package main

import (
	"fmt"

	transcribe "github.com/handy-computer/transcribe.cpp/bindings/go"
	"github.com/handy-computer/transcribe.cpp/bindings/go/internal/support"
)

func main() {
	// Loading a model would do this anyway; it is explicit here because
	// discovering devices before any load is the point of this example, and
	// a dynamic-backend build has nothing registered until it runs.
	if err := transcribe.InitBackends(""); err != nil {
		support.Die(err)
	}

	devices, err := transcribe.Devices()
	if err != nil {
		support.Die(err)
	}
	fmt.Println("discovered devices:")
	for i, d := range devices {
		fmt.Printf("  %d %-6s %-8s %s (%s), %d MB\n", i, d.Type, d.Kind, d.Name, d.Description, d.MemoryTotal>>20)
	}

	backends := []struct {
		name string
		b    transcribe.Backend
	}{
		{"cpu", transcribe.BackendCPU},
		{"metal", transcribe.BackendMetal},
		{"vulkan", transcribe.BackendVulkan},
		{"cuda", transcribe.BackendCUDA},
		{"rocm", transcribe.BackendROCm},
	}
	fmt.Println("\nbackend availability:")
	for _, b := range backends {
		fmt.Printf("  %-7s %t\n", b.name, b.b.Available())
	}

	model := support.Model(support.Arg(0), support.ModelEnv)
	if model == "" {
		support.Skip("\nno model, so device discovery only (set TRANSCRIBE_SMOKE_MODEL to load one)")
	}

	// Prefer an accelerator, and fall back to CPU on a clean failure rather
	// than probing every possibility up front.
	preferred := transcribe.BackendCPU
	name := "cpu"
	for _, b := range backends[1:] {
		if b.b.Available() {
			preferred, name = b.b, b.name
			break
		}
	}
	fmt.Printf("\nrequesting backend: %s\n", name)

	m, err := transcribe.LoadModel(model, &transcribe.LoadOptions{Backend: preferred})
	if err != nil {
		fmt.Printf("  %s unavailable (%v), retrying on cpu\n", name, err)
		m, err = transcribe.LoadModel(model, &transcribe.LoadOptions{Backend: transcribe.BackendCPU})
	}
	if err != nil {
		support.Die(err)
	}
	defer m.Close()

	dev, err := m.Device()
	if err != nil {
		support.Die(err)
	}
	fmt.Printf("loaded on backend %s, device %s (%s)\n", m.Backend(), dev.Name, dev.Type)
}
