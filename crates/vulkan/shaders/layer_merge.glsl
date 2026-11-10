// The wire form of the partial results of a tensor split, shared by
// `layer_partial_to_wire.comp` and `layer_merge_wire.comp`. It is the CUDA layer's
// (`kernels/layer_merge.cuh`), byte for byte:
//
// A row crosses between devices as `d` f16 values and one f32 factor: the row is multiplied
// by a power of two before it is rounded, so that its largest magnitude is below 32768, and
// the factor puts it back. The factor is 1 for every row that fits f16 as it is. The wire
// form of `n_rows` rows: the rows' f16 values, then (at the next multiple of 4 bytes) one
// f32 factor per row. The *wide* form carries the rows as f32 instead (and factors of 1 in
// the same place behind them).
#include "layer_common.glsl"

// Byte offset of the factors behind the values of `n_rows` rows.
uint64_t wire_factors(uint n_rows, uint d, bool wide) {
    const uint64_t values = uint64_t(n_rows) * d;
    return wide ? values * 4 : (values * 2 + 3) & ~uint64_t(3);
}

// The powers of two for a row whose largest magnitude is `m`: `down` brings the row below
// 32768, `up` puts it back. Both are 1 unless the row would overflow f16. A row that is
// not finite gets a pair that keeps it not finite.
void wire_scale(float m, out float down, out float up) {
    // m = f * 2^e with f in [0.5, 1).
    const int e = int((floatBitsToUint(m) >> 23) & 0xFFu) - 126;
    const int k = e > 15 ? e - 15 : 0;
    down = uintBitsToFloat(uint(127 - k) << 23);
    up = uintBitsToFloat(uint(127 + k) << 23);
}

// Chunk `chunk` (4 values) of a wire form whose row has the factor `up`.
vec4 wire_load(uint64_t wire, uint64_t chunk, float up, bool wide) {
    if (wide) {
        return ldv4(wire + 16 * chunk);
    }
    const uvec2 bits = LHalf4(wire + 8 * chunk).v[0];
    return vec4(unpackHalf2x16(bits.x), unpackHalf2x16(bits.y)) * up;
}

float max4(vec4 v) {
    const vec4 a = abs(v);
    return max(max(a.x, a.y), max(a.z, a.w));
}
