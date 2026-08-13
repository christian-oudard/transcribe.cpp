package transcribe

// cgo is not available in test files, so these exercise the binding through
// its own surface rather than reaching for C identifiers.

import (
	"errors"
	"strings"
	"testing"
)

func TestVersion(t *testing.T) {
	if Version() == "" {
		t.Fatal("empty version")
	}
	t.Logf("libtranscribe %s (%s)", Version(), VersionCommit())
}

func TestABICheck(t *testing.T) {
	if err := ABICheck(); err != nil {
		t.Fatal(err)
	}
}

// TestVersionGate covers the load-time check that the linked library is the
// one this binding was built for. Pre-1.0 the ABI can break between minor
// releases, and a change that keeps every struct size is invisible to
// ABICheck.
func TestVersionGate(t *testing.T) {
	if err := ensureCompatible(); err != nil {
		t.Fatalf("the library we linked is not the one we compiled against: %v", err)
	}
	if got := baseVersion(CompiledVersion); got != CompiledVersion {
		t.Errorf("CompiledVersion %q is not a bare release version (base %q)", CompiledVersion, got)
	}
	// A packaging suffix is not a different release.
	for _, c := range []struct{ in, want string }{
		{"0.2.0", "0.2.0"},
		{"0.2.0.post1", "0.2.0"},
		{"0.2.0-rc1", "0.2.0"},
		{"0.2.0+build5", "0.2.0"},
	} {
		if got := baseVersion(c.in); got != c.want {
			t.Errorf("baseVersion(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestStatusIsError(t *testing.T) {
	var err error = ErrBackend
	if !errors.Is(err, ErrBackend) {
		t.Fatalf("errors.Is failed on %v", err)
	}
	if errors.Is(err, ErrOOM) {
		t.Fatal("matched an unrelated status")
	}
	if err.Error() == "" {
		t.Fatal("empty status text")
	}
	if check(0) != nil {
		t.Fatal("OK became an error")
	}
}

// TestDevices expects a CPU device: ggml always registers one, so an empty
// list means no backend registered at all.
func TestDevices(t *testing.T) {
	devs, err := Devices()
	if err != nil {
		t.Fatal(err)
	}
	if len(devs) == 0 {
		t.Fatal("no compute devices registered")
	}
	var cpu bool
	for _, d := range devs {
		t.Logf("%s (%s, %s) %d MB", d.Name, d.Kind, d.Type, d.MemoryTotal>>20)
		if d.Type == DeviceCPU {
			cpu = true
		}
	}
	if !cpu {
		t.Error("no CPU device")
	}
	if !BackendAuto.Available() {
		t.Error("auto backend unavailable with devices registered")
	}
	if !BackendCPU.Available() {
		t.Error("CPU backend unavailable with a CPU device")
	}
}

func TestLoadMissingFile(t *testing.T) {
	m, err := LoadModel("./no-such-model.gguf", nil)
	if err == nil {
		m.Close()
		t.Fatal("loaded a model that does not exist")
	}
	if !errors.Is(err, ErrFileNotFound) {
		t.Fatalf("want ErrFileNotFound, got %v", err)
	}
}

// TestModeTables checks the three C toggle enums are covered for every Mode,
// since one Go type stands in for all three and a table with a hole would
// send the wrong value rather than fail.
func TestModeTables(t *testing.T) {
	for _, m := range []Mode{ModeDefault, ModeOff, ModeOn} {
		if int(m) >= len(pncToC) || int(m) >= len(itnToC) || int(m) >= len(diarizeToC) {
			t.Fatalf("mode %d is outside one of the tables", m)
		}
	}
	// Distinct modes must stay distinct through the tables.
	if pncToC[ModeOff] == pncToC[ModeOn] || itnToC[ModeOff] == itnToC[ModeOn] ||
		diarizeToC[ModeOff] == diarizeToC[ModeOn] {
		t.Error("a table maps off and on to the same C value")
	}
}

func TestTimestampsMapping(t *testing.T) {
	// The Go zero value has to be the library default, so the enums are
	// mapped rather than shared. Check the round trip both ways.
	for _, want := range []Timestamps{StampsAuto, StampsNone, StampsSegment, StampsWord, StampsToken} {
		if got := stampsFromC(stampsToC[want]); got != want {
			t.Errorf("%v round-tripped to %v", want, got)
		}
	}
	if StampsAuto != 0 {
		t.Error("the zero value is not the library default")
	}
}

// TestRunParamsDefaults checks that a nil RunOptions and a zero RunOptions
// build the same params, so the zero value really is "library defaults".
func TestRunParamsDefaults(t *testing.T) {
	def, freeDef := runParams(nil)
	defer freeDef()
	zero, freeZero := runParams(&RunOptions{})
	defer freeZero()
	if def != zero {
		t.Errorf("zero options differ from defaults:\n %+v\n %+v", def, zero)
	}
}

func TestLogHandler(t *testing.T) {
	var got []string
	SetLogHandler(func(_ LogLevel, msg string) { got = append(got, msg) })
	defer SetLogHandler(nil)

	// A failed load logs; nothing else here is guaranteed to.
	if _, err := LoadModel("./no-such-model.gguf", nil); err == nil {
		t.Fatal("expected a load failure")
	}
	if len(got) == 0 {
		t.Skip("library logged nothing for a missing file")
	}
	if strings.TrimSpace(strings.Join(got, "")) == "" {
		t.Error("log handler got only blank messages")
	}
}
