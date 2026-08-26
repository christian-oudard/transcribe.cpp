// arch/titanet/capabilities.cpp - TitaNet capability defaults.
//
// Applied before transcribe::read_capability_kv (KV present overrides, KV
// absent keeps the default). TitaNet produces neither text nor speaker
// segments: it answers "whose voice is this" as a vector, and something else
// has to decide what to do with it. So every transcript-shaped capability is
// off, including diarization -- a model that cannot say when anyone spoke
// does not diarize, whatever it is used to build.

#include "titanet.h"

namespace transcribe::titanet {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    // Fixed 16 kHz mel bank (NeMo AudioToMelSpectrogramPreprocessor).
    caps.native_sample_rate = 16000;

    caps.supports_translate = false;
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // The rows come from clustering the embeddings, so they are as good as the
    // windowing in front of them; see src/arch/titanet/diarize.cpp.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_DIARIZATION, true);

    // Abort callback honored at the top of each run.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::titanet
