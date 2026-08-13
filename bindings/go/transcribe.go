// Package transcribe is a Go binding for transcribe.cpp, a ggml speech
// recognition library covering whisper, parakeet, canary and other families
// behind one C API.
//
// The binding links against an installed libtranscribe. cgo has no build
// script, so point it at the install prefix through the environment:
//
//	export CGO_CFLAGS="-I$PREFIX/include"
//	export CGO_LDFLAGS="-L$PREFIX/lib64"
//	export LD_LIBRARY_PATH="$PREFIX/lib64"
//
// A typical run loads a model, opens a session against it and transcribes
// 16 kHz mono float32 PCM:
//
//	s, err := transcribe.Open("parakeet.gguf", nil, nil)
//	if err != nil {
//		return err
//	}
//	defer s.Close()
//
//	res, err := s.Run(ctx, pcm, nil)
//	if err != nil {
//		return err
//	}
//	fmt.Println(res.Text)
//
// Handles are freed by Close, not by a finalizer: a Model must outlive every
// Session made from it, an ordering the garbage collector does not know
// about. Results own their contents, so they outlive both.
//
// A Session is single-threaded. Compute is more restricted still: the library
// allows at most one run, batch or active stream in flight across every
// session of a given model, so transcribing in parallel means one Model per
// worker. See the README's threading section.
package transcribe

/*
#cgo LDFLAGS: -ltranscribe

#include <stdlib.h>
#include <transcribe.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

// CompiledVersion is the library version this binding was written against.
// Kept in step with include/transcribe.h by the version-sync CI gate.
const CompiledVersion = "0.2.0"

// Version is the linked library's version string, e.g. "0.2.0".
func Version() string { return C.GoString(C.transcribe_version()) }

// VersionCommit is the git commit the library was built from.
func VersionCommit() string { return C.GoString(C.transcribe_version_commit()) }

// ErrVersionMismatch is what a load returns when the linked library is not
// the one this binding was built for.
var ErrVersionMismatch = errors.New("transcribe: library and binding versions differ")

// baseVersion is the leading dotted-numeric release segment, so a packaging
// suffix such as "0.2.0.post1" compares equal to "0.2.0".
func baseVersion(v string) string {
	end := strings.IndexFunc(v, func(r rune) bool {
		return !(r >= '0' && r <= '9') && r != '.'
	})
	if end >= 0 {
		v = v[:end]
	}
	return strings.TrimRight(v, ".")
}

// ensureCompatible refuses a library whose base version differs from the one
// this binding was built against. Pre-1.0 the ABI may break between minor
// releases, and the struct-size check cannot see a change that keeps every
// size, such as a renumbered enum. The abihash pin catches those at build
// time; this catches them at run time, when a stale shared library is what
// actually got loaded.
func ensureCompatible() error {
	if have, want := baseVersion(Version()), baseVersion(CompiledVersion); have != want {
		return fmt.Errorf("%w: linked %s, built for %s", ErrVersionMismatch, Version(), CompiledVersion)
	}
	return nil
}

// Status is a library return code. A non-OK status is returned as an error,
// so callers match one with errors.Is:
//
//	if errors.Is(err, transcribe.ErrBackend) { ... }
type Status int

const (
	ErrInvalidArg          Status = C.TRANSCRIBE_ERR_INVALID_ARG
	ErrNotImplemented      Status = C.TRANSCRIBE_ERR_NOT_IMPLEMENTED
	ErrFileNotFound        Status = C.TRANSCRIBE_ERR_FILE_NOT_FOUND
	ErrGGUF                Status = C.TRANSCRIBE_ERR_GGUF
	ErrUnsupportedArch     Status = C.TRANSCRIBE_ERR_UNSUPPORTED_ARCH
	ErrUnsupportedVariant  Status = C.TRANSCRIBE_ERR_UNSUPPORTED_VARIANT
	ErrOOM                 Status = C.TRANSCRIBE_ERR_OOM
	ErrBackend             Status = C.TRANSCRIBE_ERR_BACKEND
	ErrSampleRate          Status = C.TRANSCRIBE_ERR_SAMPLE_RATE
	ErrUnsupportedLanguage Status = C.TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE
	ErrUnsupportedTask     Status = C.TRANSCRIBE_ERR_UNSUPPORTED_TASK
	ErrUnsupportedStamps   Status = C.TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS
	ErrAborted             Status = C.TRANSCRIBE_ERR_ABORTED
	ErrBadStructSize       Status = C.TRANSCRIBE_ERR_BAD_STRUCT_SIZE
	ErrUnsupportedPNC      Status = C.TRANSCRIBE_ERR_UNSUPPORTED_PNC
	ErrUnsupportedITN      Status = C.TRANSCRIBE_ERR_UNSUPPORTED_ITN
	ErrInputTooLong        Status = C.TRANSCRIBE_ERR_INPUT_TOO_LONG
	ErrOutputTruncated     Status = C.TRANSCRIBE_ERR_OUTPUT_TRUNCATED
)

// Error makes Status usable as an error; the text comes from the library, so
// a status this binding predates still describes itself.
func (s Status) Error() string {
	return C.GoString(C.transcribe_status_string(C.int(s)))
}

// check turns a status into an error, with OK becoming nil.
func check(s C.transcribe_status) error {
	if s == C.TRANSCRIBE_OK {
		return nil
	}
	return Status(s)
}

// Backend is a request for where compute should run. Auto probes for the
// fastest available; the rest require that backend and fail with ErrBackend
// when the library was not built with it or no such device is present.
type Backend int

const (
	BackendAuto     Backend = C.TRANSCRIBE_BACKEND_AUTO
	BackendCPU      Backend = C.TRANSCRIBE_BACKEND_CPU
	BackendMetal    Backend = C.TRANSCRIBE_BACKEND_METAL
	BackendVulkan   Backend = C.TRANSCRIBE_BACKEND_VULKAN
	BackendCPUAccel Backend = C.TRANSCRIBE_BACKEND_CPU_ACCEL
	BackendCUDA     Backend = C.TRANSCRIBE_BACKEND_CUDA
	BackendROCm     Backend = C.TRANSCRIBE_BACKEND_ROCM
)

// Available reports whether some registered device can satisfy this request,
// which turns an impossible backend choice into a clear error before a model
// load rather than after it.
func (b Backend) Available() bool {
	return bool(C.transcribe_backend_available(C.transcribe_backend_request(b)))
}

// DeviceType is the CPU/GPU/iGPU/accelerator axis, orthogonal to the vendor
// in Device.Kind. Backends classify themselves, so treat it as a placement
// hint rather than a hardware taxonomy.
type DeviceType int

const (
	DeviceCPU   DeviceType = C.TRANSCRIBE_DEVICE_TYPE_CPU
	DeviceGPU   DeviceType = C.TRANSCRIBE_DEVICE_TYPE_GPU
	DeviceIGPU  DeviceType = C.TRANSCRIBE_DEVICE_TYPE_IGPU
	DeviceAccel DeviceType = C.TRANSCRIBE_DEVICE_TYPE_ACCEL
)

func (t DeviceType) String() string {
	switch t {
	case DeviceCPU:
		return "cpu"
	case DeviceGPU:
		return "gpu"
	case DeviceIGPU:
		return "igpu"
	case DeviceAccel:
		return "accel"
	}
	return "unknown"
}

// Device is one registered compute device.
type Device struct {
	// Name is ggml's name for it, Description the human-readable one.
	Name        string
	Description string
	// Kind is the vendor classification: cpu, accel, metal, vulkan, cuda,
	// rocm, sycl, gpu, unknown.
	Kind string
	// ID is a stable hardware identifier, the PCI bus id for PCI devices,
	// or empty when the backend reports none.
	ID   string
	Type DeviceType
	// MemoryTotal is the reported capacity in bytes, 0 when unreported.
	MemoryTotal uint64
	// MemoryFree is a snapshot taken when this struct was filled: it is
	// stale as soon as anything allocates. Call Devices again to refresh.
	MemoryFree uint64
}

// DeviceCount is how many compute devices are registered.
func DeviceCount() int { return int(C.transcribe_backend_device_count()) }

// getDevice returns the device at index, which is stable for the life of the
// process, so the same index always names the same device.
func getDevice(index int) (Device, error) {
	var cd C.struct_transcribe_backend_device
	C.transcribe_backend_device_init(&cd)
	if err := check(C.transcribe_get_backend_device(C.int(index), &cd)); err != nil {
		return Device{}, err
	}
	return goDevice(&cd), nil
}

// Devices lists every registered compute device. Its index is what
// LoadOptions.GPUDevice selects by.
func Devices() ([]Device, error) {
	n := DeviceCount()
	out := make([]Device, 0, n)
	for i := range n {
		d, err := getDevice(i)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, nil
}

// goDevice copies a device struct out of library-owned storage. Every string
// on it is a borrowed pointer valid for the life of the process, but copying
// keeps the Go type free of unsafe.Pointer.
func goDevice(cd *C.struct_transcribe_backend_device) Device {
	return Device{
		Name:        C.GoString(cd.name),
		Description: C.GoString(cd.description),
		Kind:        C.GoString(cd.kind),
		ID:          C.GoString(cd.device_id),
		Type:        DeviceType(cd.device_type),
		MemoryTotal: uint64(cd.memory_total),
		MemoryFree:  uint64(cd.memory_free),
	}
}

// abiStruct is one public struct's layout as this binding sees it, paired
// with the library's id for it.
type abiStruct struct {
	name  string
	which C.transcribe_abi_struct
	size  uintptr
	align uintptr
}

// abiStructs is every struct the library reports a layout for and this
// binding allocates.
var abiStructs = []abiStruct{
	{"model_load_params", C.TRANSCRIBE_ABI_MODEL_LOAD_PARAMS,
		unsafe.Sizeof(C.struct_transcribe_model_load_params{}), unsafe.Alignof(C.struct_transcribe_model_load_params{})},
	{"session_params", C.TRANSCRIBE_ABI_SESSION_PARAMS,
		unsafe.Sizeof(C.struct_transcribe_session_params{}), unsafe.Alignof(C.struct_transcribe_session_params{})},
	{"run_params", C.TRANSCRIBE_ABI_RUN_PARAMS,
		unsafe.Sizeof(C.struct_transcribe_run_params{}), unsafe.Alignof(C.struct_transcribe_run_params{})},
	{"capabilities", C.TRANSCRIBE_ABI_CAPABILITIES,
		unsafe.Sizeof(C.struct_transcribe_capabilities{}), unsafe.Alignof(C.struct_transcribe_capabilities{})},
	{"timings", C.TRANSCRIBE_ABI_TIMINGS,
		unsafe.Sizeof(C.struct_transcribe_timings{}), unsafe.Alignof(C.struct_transcribe_timings{})},
	{"segment", C.TRANSCRIBE_ABI_SEGMENT,
		unsafe.Sizeof(C.struct_transcribe_segment{}), unsafe.Alignof(C.struct_transcribe_segment{})},
	{"word", C.TRANSCRIBE_ABI_WORD,
		unsafe.Sizeof(C.struct_transcribe_word{}), unsafe.Alignof(C.struct_transcribe_word{})},
	{"token", C.TRANSCRIBE_ABI_TOKEN,
		unsafe.Sizeof(C.struct_transcribe_token{}), unsafe.Alignof(C.struct_transcribe_token{})},
	{"session_limits", C.TRANSCRIBE_ABI_SESSION_LIMITS,
		unsafe.Sizeof(C.struct_transcribe_session_limits{}), unsafe.Alignof(C.struct_transcribe_session_limits{})},
	{"backend_device", C.TRANSCRIBE_ABI_BACKEND_DEVICE,
		unsafe.Sizeof(C.struct_transcribe_backend_device{}), unsafe.Alignof(C.struct_transcribe_backend_device{})},
	{"speaker_segment", C.TRANSCRIBE_ABI_SPEAKER_SEGMENT,
		unsafe.Sizeof(C.struct_transcribe_speaker_segment{}), unsafe.Alignof(C.struct_transcribe_speaker_segment{})},
}

// ABICheck reports whether this binding's view of the library's public
// structs is safe to use against the library it is linked to. LoadModel and
// Open run it once before the first load, so calling it directly is only
// worth it to fail earlier, or to report the mismatch differently.
//
// Every params struct is filled by a library _init() that stamps its own
// sizeof into the caller's buffer, so a buffer smaller than the library's
// struct gets written past the end, and one aligned more loosely than the
// library expects is misaligned for the fields it writes. The layouts here
// come from the header the binding was compiled against, so a mismatch means
// that header and the shared library have drifted apart, which cgo normally
// rules out but a stale library on LD_LIBRARY_PATH does not. The library
// grows structs only at the end, so a binding built against a newer header
// than the library has larger values, which is safe and reports no error.
func ABICheck() error {
	var bad []string
	for _, s := range abiStructs {
		size := uintptr(C.transcribe_abi_struct_size(s.which))
		align := uintptr(C.transcribe_abi_struct_align(s.which))
		switch {
		case size == 0:
			bad = append(bad, fmt.Sprintf("%s: library reports no layout for it", s.name))
		case s.size < size:
			bad = append(bad, fmt.Sprintf("%s: binding has %d bytes, library writes %d", s.name, s.size, size))
		case s.align < align:
			bad = append(bad, fmt.Sprintf("%s: binding aligns to %d, library expects %d", s.name, s.align, align))
		}
	}
	if bad != nil {
		return fmt.Errorf("transcribe ABI mismatch with libtranscribe %s: %s",
			Version(), strings.Join(bad, "; "))
	}
	return nil
}

var abiOnce struct {
	sync.Once
	err error
}

// abiGate runs ABICheck once per process, before the first model is loaded.
// Every later load repeats the verdict without repeating the work.
func abiGate() error {
	abiOnce.Do(func() { abiOnce.err = ABICheck() })
	return abiOnce.err
}

var backendsOnce struct {
	sync.Once
	err error
}

// InitBackends loads ggml's compute backends from dir. It matters only to
// builds that ship backends as separate shared libraries (GGML_BACKEND_DL);
// with them compiled in it is a no-op. An empty dir uses the library's own
// default, the directory holding libtranscribe itself.
//
// Calling this is optional: the first model load does it with an empty dir,
// so a binding works in both postures untouched. Only a build that keeps its
// backend modules somewhere else needs to call it, and then it has to be
// before the first load, since the first call is the one that counts.
func InitBackends(dir string) error {
	backendsOnce.Do(func() { backendsOnce.err = initBackends(dir) })
	return backendsOnce.err
}

func initBackends(dir string) error {
	if dir == "" {
		return check(C.transcribe_init_backends_default())
	}
	cdir := C.CString(dir)
	defer C.free(unsafe.Pointer(cdir))
	return check(C.transcribe_init_backends(cdir))
}

// ready runs the once-per-process startup that has to happen before the first
// model load: check the version, then the ABI, then register the compute
// backends.
func ready() error {
	if err := ensureCompatible(); err != nil {
		return err
	}
	if err := abiGate(); err != nil {
		return err
	}
	return InitBackends("")
}
