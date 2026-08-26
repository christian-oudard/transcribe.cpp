// arch/fsmn_vad/capabilities.cpp - FSMN VAD capability defaults.

#include "fsmn_vad.h"

namespace transcribe::fsmn_vad {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;
    caps.supports_translate = false;
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // The rows this produces are speech regions with one anonymous speaker,
    // published on the diarization surface because that is the surface for
    // "who spoke when" and this answers the "when" half of it. Nothing here
    // tells two voices apart, and a caller that needs that runs a diarizer
    // over these regions rather than instead of them.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_DIARIZATION, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::fsmn_vad
