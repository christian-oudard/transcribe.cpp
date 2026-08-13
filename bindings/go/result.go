package transcribe

/*
#include <transcribe.h>
*/
import "C"

import "time"

// Result is one transcription, fully materialized. Every string and row is
// copied out of session-owned storage when the result is made, so a Result
// stays valid after the next run and after the session is closed.
//
// This is the contract every binding in this repo keeps: results own their
// rows. The library's own accessors hand back borrowed pointers that the next
// run replaces, so nothing here points into the session.
type Result struct {
	// Text is the transcript, cleaned of control tags.
	Text string
	// RawText is the decode before family post-processing, with every
	// special tag the model emitted still in place. Unlike
	// RunOptions.KeepSpecialTags it costs nothing and does not replace Text.
	// Offline runs only; a stream snapshot leaves it empty.
	RawText string
	// Language is the language code the model detected, or empty when it
	// does not detect one.
	Language string
	// Timestamps is the granularity the run actually produced, which is what
	// an Auto request resolved to.
	Timestamps Timestamps

	Segments []Segment
	// SpeakerSegments is the speaker-activity view, populated only when the
	// run resolved diarization on.
	SpeakerSegments []SpeakerSegment
	// Words is populated when the run produced word timestamps, Tokens when
	// it produced token timestamps.
	Words  []Word
	Tokens []Token

	Timings Timings
}

// Segment is one span of transcript. First and count fields index into the
// Words and Tokens slices of the same result.
type Segment struct {
	Start, End time.Duration
	FirstWord  int
	NumWords   int
	FirstToken int
	NumTokens  int
	Text       string
	// Speaker is a 1-based id when the run resolved diarization on, 0
	// otherwise.
	Speaker int
}

// Word is one word with its span.
type Word struct {
	Start, End time.Duration
	Segment    int
	FirstToken int
	NumTokens  int
	Text       string
}

// Token is one model token.
type Token struct {
	ID int
	// P is a confidence hint, or NaN when the family produces none. What it
	// means varies by architecture, so do not read it as a calibrated
	// probability.
	P          float32
	Start, End time.Duration
	Segment    int
	Word       int
	Text       string
}

// SpeakerSegment is one "who spoke when" row, the view of speaker activity
// that does not depend on the transcript. Rows may overlap in time when two
// people talk at once. A model that attributes text but carries no timing
// reports a zero span.
type SpeakerSegment struct {
	Start, End time.Duration
	Speaker    int
	// P is the attribution confidence, or NaN when the model gives none.
	P float32
}

// Timings is where a run spent its time. Load is the model load, which is
// non-zero from the moment the model is loaded; the rest are per run.
type Timings struct {
	Load   time.Duration
	Mel    time.Duration
	Encode time.Duration
	Decode time.Duration
}

// ms turns the library's millisecond stamps into a Duration.
func ms(v C.int64_t) time.Duration { return time.Duration(v) * time.Millisecond }

// reader pulls one result out of a session's borrowed storage. The library
// splits every accessor into a single-run and a batch form, so this holds the
// utterance index, -1 for a single run, and branches once per accessor.
//
// It exists only long enough to fill a Result; nothing it returns outlives
// the next run.
type reader struct {
	s *Session
	i int
}

func (r reader) batch() bool { return r.i >= 0 }

// status is the per-utterance verdict within a batch. One utterance can fail
// while the batch as a whole succeeds.
func (r reader) status() error {
	if !r.batch() {
		return nil
	}
	return check(C.transcribe_batch_status(r.s.c, C.int(r.i)))
}

// materialize copies the whole result into Go memory.
func (r reader) materialize() Result {
	return Result{
		Text:            r.text(),
		RawText:         r.rawText(),
		Language:        r.language(),
		Timestamps:      r.timestamps(),
		Segments:        r.segments(),
		SpeakerSegments: r.speakerSegments(),
		Words:           r.words(),
		Tokens:          r.tokens(),
		Timings:         r.timings(),
	}
}

func (r reader) text() string {
	if r.batch() {
		return C.GoString(C.transcribe_batch_full_text(r.s.c, C.int(r.i)))
	}
	return C.GoString(C.transcribe_full_text(r.s.c))
}

func (r reader) rawText() string {
	if r.batch() {
		return C.GoString(C.transcribe_batch_raw_text(r.s.c, C.int(r.i)))
	}
	return C.GoString(C.transcribe_raw_text(r.s.c))
}

func (r reader) language() string {
	if r.batch() {
		return C.GoString(C.transcribe_batch_detected_language(r.s.c, C.int(r.i)))
	}
	return C.GoString(C.transcribe_detected_language(r.s.c))
}

func (r reader) timestamps() Timestamps {
	if r.batch() {
		return stampsFromC(C.transcribe_batch_returned_timestamp_kind(r.s.c, C.int(r.i)))
	}
	return stampsFromC(C.transcribe_returned_timestamp_kind(r.s.c))
}

func (r reader) segments() []Segment {
	var n int
	if r.batch() {
		n = int(C.transcribe_batch_n_segments(r.s.c, C.int(r.i)))
	} else {
		n = int(C.transcribe_n_segments(r.s.c))
	}
	if n <= 0 {
		return nil
	}
	// One struct for the whole loop: the getters keep the struct_size the
	// init stamped and zero the rest before writing, so no row can see the
	// row before it.
	out := make([]Segment, n)
	var c C.struct_transcribe_segment
	C.transcribe_segment_init(&c)
	for j := range out {
		if r.batch() {
			C.transcribe_batch_get_segment(r.s.c, C.int(r.i), C.int(j), &c)
		} else {
			C.transcribe_get_segment(r.s.c, C.int(j), &c)
		}
		out[j] = Segment{
			Start:      ms(c.t0_ms),
			End:        ms(c.t1_ms),
			FirstWord:  int(c.first_word),
			NumWords:   int(c.n_words),
			FirstToken: int(c.first_token),
			NumTokens:  int(c.n_tokens),
			Text:       C.GoString(c.text),
			Speaker:    int(c.speaker_id),
		}
	}
	return out
}

func (r reader) words() []Word {
	var n int
	if r.batch() {
		n = int(C.transcribe_batch_n_words(r.s.c, C.int(r.i)))
	} else {
		n = int(C.transcribe_n_words(r.s.c))
	}
	if n <= 0 {
		return nil
	}
	out := make([]Word, n)
	var c C.struct_transcribe_word
	C.transcribe_word_init(&c)
	for j := range out {
		if r.batch() {
			C.transcribe_batch_get_word(r.s.c, C.int(r.i), C.int(j), &c)
		} else {
			C.transcribe_get_word(r.s.c, C.int(j), &c)
		}
		out[j] = Word{
			Start:      ms(c.t0_ms),
			End:        ms(c.t1_ms),
			Segment:    int(c.seg_index),
			FirstToken: int(c.first_token),
			NumTokens:  int(c.n_tokens),
			Text:       C.GoString(c.text),
		}
	}
	return out
}

func (r reader) tokens() []Token {
	var n int
	if r.batch() {
		n = int(C.transcribe_batch_n_tokens(r.s.c, C.int(r.i)))
	} else {
		n = int(C.transcribe_n_tokens(r.s.c))
	}
	if n <= 0 {
		return nil
	}
	out := make([]Token, n)
	var c C.struct_transcribe_token
	C.transcribe_token_init(&c)
	for j := range out {
		if r.batch() {
			C.transcribe_batch_get_token(r.s.c, C.int(r.i), C.int(j), &c)
		} else {
			C.transcribe_get_token(r.s.c, C.int(j), &c)
		}
		out[j] = Token{
			ID:      int(c.id),
			P:       float32(c.p),
			Start:   ms(c.t0_ms),
			End:     ms(c.t1_ms),
			Segment: int(c.seg_index),
			Word:    int(c.word_index),
			Text:    C.GoString(c.text),
		}
	}
	return out
}

func (r reader) speakerSegments() []SpeakerSegment {
	var n int
	if r.batch() {
		n = int(C.transcribe_batch_n_speaker_segments(r.s.c, C.int(r.i)))
	} else {
		n = int(C.transcribe_n_speaker_segments(r.s.c))
	}
	if n <= 0 {
		return nil
	}
	out := make([]SpeakerSegment, n)
	var c C.struct_transcribe_speaker_segment
	C.transcribe_speaker_segment_init(&c)
	for j := range out {
		if r.batch() {
			C.transcribe_batch_get_speaker_segment(r.s.c, C.int(r.i), C.int(j), &c)
		} else {
			C.transcribe_get_speaker_segment(r.s.c, C.int(j), &c)
		}
		out[j] = SpeakerSegment{
			Start:   ms(c.t0_ms),
			End:     ms(c.t1_ms),
			Speaker: int(c.speaker_id),
			P:       float32(c.p),
		}
	}
	return out
}

// timings fills the timing block. A failure here can only mean the session or
// index is bad, which the caller has already been told about through the run's
// own status, so the zeroed struct stands in rather than growing an error
// return on every accessor.
func (r reader) timings() Timings {
	var c C.struct_transcribe_timings
	C.transcribe_timings_init(&c)
	if r.batch() {
		C.transcribe_batch_get_timings(r.s.c, C.int(r.i), &c)
	} else {
		C.transcribe_get_timings(r.s.c, &c)
	}
	fms := func(v C.float) time.Duration {
		return time.Duration(float64(v) * float64(time.Millisecond))
	}
	return Timings{
		Load:   fms(c.load_ms),
		Mel:    fms(c.mel_ms),
		Encode: fms(c.encode_ms),
		Decode: fms(c.decode_ms),
	}
}
