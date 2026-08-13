package transcribe

// The library calls back into Go for cancellation and logging. cgo forbids a
// preamble with definitions in a file that uses //export, so this file's
// preamble only declares the trampolines; their bodies are the Go functions
// below, which cgo emits with C linkage.

/*
#include <transcribe.h>

extern bool transcribeAbortTrampoline(void * user_data);
// cgo drops const from an exported Go function's signature, so this
// declares char * and the cast at the call site restores the callback type.
extern void transcribeLogTrampoline(transcribe_log_level level, char * msg, void * userdata);
*/
import "C"

import (
	"context"
	"runtime/cgo"
	"sync"
	"unsafe"
)

//export transcribeAbortTrampoline
func transcribeAbortTrampoline(userData unsafe.Pointer) C.bool {
	ctx := cgo.Handle(*(*C.uintptr_t)(userData)).Value().(context.Context)
	select {
	case <-ctx.Done():
		return C.bool(true)
	default:
		return C.bool(false)
	}
}

// LogLevel is the severity of a library message. The library maps ggml's own
// levels onto these before they reach a handler, so a handler only ever sees
// these values.
type LogLevel int

const (
	LogNone  LogLevel = C.TRANSCRIBE_LOG_LEVEL_NONE
	LogInfo  LogLevel = C.TRANSCRIBE_LOG_LEVEL_INFO
	LogWarn  LogLevel = C.TRANSCRIBE_LOG_LEVEL_WARN
	LogError LogLevel = C.TRANSCRIBE_LOG_LEVEL_ERROR
	LogDebug LogLevel = C.TRANSCRIBE_LOG_LEVEL_DEBUG
	// LogCont marks a fragment continuing the previous message, which is
	// how progress output arrives. Join these onto the message before it
	// to get one line per message.
	LogCont LogLevel = C.TRANSCRIBE_LOG_LEVEL_CONT
)

func (l LogLevel) String() string {
	switch l {
	case LogNone:
		return "none"
	case LogInfo:
		return "info"
	case LogWarn:
		return "warn"
	case LogError:
		return "error"
	case LogDebug:
		return "debug"
	case LogCont:
		return "cont"
	}
	return "unknown"
}

var logHandler struct {
	sync.RWMutex
	f func(LogLevel, string)
}

// SetLogHandler routes every library and ggml message to f. A nil f drops
// them instead. Until this is called they go to stderr, and there is no way
// back to that, so call it once at startup or not at all.
//
// Messages arrive on whichever thread produced them, including inside a run.
func SetLogHandler(f func(level LogLevel, msg string)) {
	logHandler.Lock()
	logHandler.f = f
	logHandler.Unlock()
	if f == nil {
		// A null callback is the library's "drop everything" state,
		// distinct from never having called it.
		C.transcribe_log_set(nil, nil)
		return
	}
	C.transcribe_log_set(C.transcribe_log_callback(C.transcribeLogTrampoline), nil)
}

//export transcribeLogTrampoline
func transcribeLogTrampoline(level C.transcribe_log_level, msg *C.char, _ unsafe.Pointer) {
	logHandler.RLock()
	f := logHandler.f
	logHandler.RUnlock()
	if f != nil {
		f(LogLevel(level), C.GoString(msg))
	}
}
