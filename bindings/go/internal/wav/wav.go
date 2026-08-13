// Package wav reads the mono 16-bit PCM the tests and examples transcribe.
// It handles just enough of the format for the samples in this repo; the
// library itself takes float32 and links no decoder.
package wav

import (
	"encoding/binary"
	"fmt"
	"os"
)

// SampleRate is the only rate the library accepts.
const SampleRate = 16000

// Read returns the file's samples scaled to the float32 in [-1, 1] the
// library wants.
func Read(path string) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, fmt.Errorf("%s: not a WAV file", path)
	}
	var channels, bits int
	var rate uint32
	for off := 12; off+8 <= len(raw); {
		id := string(raw[off : off+4])
		size := int(binary.LittleEndian.Uint32(raw[off+4 : off+8]))
		body := raw[off+8:]
		if size > len(body) {
			return nil, fmt.Errorf("%s: chunk %q runs past the end", path, id)
		}
		body = body[:size]
		switch id {
		case "fmt ":
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			rate = binary.LittleEndian.Uint32(body[4:8])
			bits = int(binary.LittleEndian.Uint16(body[14:16]))
		case "data":
			if channels != 1 || bits != 16 || rate != SampleRate {
				return nil, fmt.Errorf("%s: want mono 16-bit %d Hz, got %d ch %d-bit %d Hz",
					path, SampleRate, channels, bits, rate)
			}
			pcm := make([]float32, len(body)/2)
			for i := range pcm {
				pcm[i] = float32(int16(binary.LittleEndian.Uint16(body[2*i:]))) / 32768
			}
			return pcm, nil
		}
		// Chunks are word-aligned, so an odd size carries a pad byte.
		off += 8 + size + size%2
	}
	return nil, fmt.Errorf("%s: no data chunk", path)
}
