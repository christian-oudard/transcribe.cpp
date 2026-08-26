// arch/titanet/diarize.cpp - who spoke when, from speaker embeddings.
//
// The pipeline TitaNet exists for. The recording is cut into short windows,
// each window becomes a vector, vectors that point the same way are the same
// voice, and runs of windows become rows.
//
// What this does not do is decide where speech is: it drops quiet windows and
// nothing more. That is enough for a recording of people talking and not
// enough for one with music, laughter or a door in it, where a real
// segmenter -- sortformer, which is already in this library -- belongs in
// front. The seam is deliberate: everything here takes windows and gives
// rows, whoever chose the windows.
//
// The cost model is one graph, computed once per window. A window is 1.5
// seconds, so an hour of speech is a few thousand computes of a small graph
// rather than one enormous one, and memory stays flat however long the
// recording is. That is the opposite of every other model here, and it is why
// diarizing a two-hour recording is possible at all.

#include "../../diarize/cluster.h"
#include "../../diarize/windows.h"
#include "ggml.h"
#include "titanet.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-log.h"

#include <limits>
#include <vector>

namespace transcribe::titanet {

namespace dz = transcribe::diarize;

transcribe_status diarize(TitanetSession * pc,
                          TitanetModel *   pm,
                          const float *    pcm,
                          int              n_samples,
                          int32_t          num_speakers,
                          float            threshold) {
    dz::WindowConfig wcfg;
    wcfg.sample_rate = pm->hparams.fe_sample_rate;

    const std::vector<dz::Window> windows = dz::speech_windows(pcm, n_samples, wcfg);
    if (windows.empty()) {
        // No speech to attribute is a legitimate answer, not a failure: a
        // recording of an empty room diarizes into nothing.
        return TRANSCRIBE_OK;
    }

    // Every window is the same length but the last, and the graph is built for
    // a fixed frame count, so the odd one out is embedded on its own graph.
    const int32_t width = windows[0].to - windows[0].from;

    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
        pc->compute_ctx = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size     = 16 * 1024 * 1024;
    ip.mem_buffer   = nullptr;
    ip.no_alloc     = true;
    pc->compute_ctx = ggml_init(ip);
    if (pc->compute_ctx == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    int                mel_n_mels = 0, T = 0;
    std::vector<float> mel;
    if (const transcribe_status st =
            pm->mel->compute(pcm + windows[0].from, static_cast<size_t>(width), mel, mel_n_mels, T, pc->n_threads);
        st != TRANSCRIBE_OK) {
        return st;
    }

    EmbedGraph g = build_embed_graph(pc->compute_ctx, *pm, T, mel_n_mels);
    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(pm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(pm->plan.scheduler_list.size()),
                                           /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) {
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, g.graph)) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    transcribe::configure_sched_n_threads(pc->sched, pc->n_threads);

    // The pooling weights every frame equally when it takes the clip
    // statistics; one buffer serves every window, since they are all this
    // long.
    const std::vector<float> even(static_cast<size_t>(T), 1.0f / static_cast<float>(T));

    const int32_t      dim = pm->hparams.embedding_size;
    std::vector<float> embeddings(windows.size() * static_cast<size_t>(dim));
    std::vector<dz::Window> embedded;
    embedded.reserve(windows.size());

    for (const dz::Window & w : windows) {
        if (pc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        if (w.to - w.from != width) {
            continue;  // The short tail; see above.
        }
        if (const transcribe_status st = pm->mel->compute(pcm + w.from, static_cast<size_t>(width), mel, mel_n_mels, T,
                                                          pc->n_threads);
            st != TRANSCRIBE_OK) {
            return st;
        }
        ggml_backend_tensor_set(g.mel_in, mel.data(), 0, mel.size() * sizeof(float));
        ggml_backend_tensor_set(g.uniform, even.data(), 0, even.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(pc->sched, g.graph) != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: graph_compute failed while diarizing");
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_backend_tensor_get(g.emb, embeddings.data() + embedded.size() * static_cast<size_t>(dim), 0,
                                static_cast<size_t>(dim) * sizeof(float));
        embedded.push_back(w);
    }
    if (embedded.empty()) {
        return TRANSCRIBE_OK;
    }

    dz::ClusterConfig ccfg;
    ccfg.num_speakers = num_speakers;
    if (threshold > 0.0f) {
        ccfg.threshold = threshold;
    }
    const std::vector<int32_t> labels =
        dz::cluster(embeddings.data(), static_cast<int32_t>(embedded.size()), dim, ccfg);

    for (const dz::Row & row : dz::rows_from_labels(embedded, labels, wcfg.sample_rate)) {
        transcribe_session::SpeakerSegmentEntry entry;
        entry.t0_ms      = row.t0_ms;
        entry.t1_ms      = row.t1_ms;
        entry.speaker_id = row.speaker + 1;  // published 1-based, as everywhere
        entry.p          = std::numeric_limits<float>::quiet_NaN();
        pc->speaker_segments.push_back(entry);
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::titanet
