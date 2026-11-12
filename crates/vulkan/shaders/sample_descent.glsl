// The exact cut-offs of top-k, top-p and typical sampling, and the hundredth token of mirostat 1:
// a radix descent over keys (`sample_hist`, `sample_export`, `sample_decide`).
//
// A token's key is the bit pattern of a distance that orders like its probability (for top-k and
// top-p the distance below the best logit; for typical sampling the distance of the scaled logit
// from its mean), so "the k most probable tokens" or "the most probable tokens that hold the
// share p" are both "the tokens whose key is at most K". K is found from the top bits down, in
// three levels of 11, 11 and 10 bits: a level counts, per value of its bits, the tokens that
// agree with the bits found so far (and sums their weights), and the first value at which the
// running total reaches the target is the next part of K. Only the tokens of the row's set
// count.
//
// A level is three shaders. `sample_hist` builds the histogram of a device's slice of the
// vocabulary, `sample_export` turns it into f32 values that add exactly (so the histograms of
// several devices are joined by the sum the devices have anyway), and `sample_decide` reads the
// total and moves the row's state one level down.
//
// Tokens tied at the cut are all kept: the cut is a threshold on the probability, not a number
// of tokens.
#include "sample_common.glsl"

// Which descent a level belongs to.
const uint DESC_TOPK = 0u;
const uint DESC_TOPP = 1u;
const uint DESC_TYPICAL = 2u;
const uint DESC_K100 = 3u;

bool desc_runs(Params p, State s, uint desc, uint position) {
    if ((p.flags & F_SAMPLE) == 0u || (s.status & S_EMPTY) != 0u || s.cursor != position) {
        return false;
    }
    const uint stage = stage_at(p, position);
    switch (desc) {
    case DESC_TOPK:
        return stage == STAGE_TOPK || stage == STAGE_MIRO1;
    case DESC_TOPP:
        return stage == STAGE_TOPP;
    case DESC_TYPICAL:
        return stage == STAGE_TYPICAL;
    default:
        return stage == STAGE_MIRO1;
    }
}

// The bins of a level: level 0 takes the top 11 bits of the key, which never exceed 1020 for a
// finite or infinite distance.
uint level_bins(uint level) { return level == 1u ? 2048u : 1024u; }
