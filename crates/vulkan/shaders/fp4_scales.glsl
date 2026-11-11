// The scales of the 4-bit float types, shared by the weight kernels (mmvq_mxfp4.glsl,
// mmvq_nvfp4.glsl, mmd_mxfp4.glsl, mmd_nvfp4.glsl) and the embedding gather
// (layer_embed_gather.comp), so that they cannot disagree about a format; the CUDA
// kernels' are in fp4_scales.cuh. Both return the scale halved, because the levels of
// the table (`ares_kvalues_mxfp4`) are doubled to make them integers.

// The scale of an MXFP4 block, 2^(e - 127) / 2, for its E8M0 byte e: e = 0 is the
// subnormal 2^-128 and e = 255, which is NaN in the format, is taken as the large finite
// 2^127, as the reference does.
float e8m0_half(uint e) {
    return uintBitsToFloat(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
}

// The scale of an NVFP4 sub-block for its UE4M3 byte x (4 exponent bits with bias 7, 3
// mantissa bits). 0 and the NaN code 0x7F give 0.
float ue4m3_half(uint x) {
    if (x == 0u || x == 0x7Fu) {
        return 0.0;
    }
    const uint e = (x >> 3) & 0xFu;
    const float man = float(x & 7u);
    const float raw = e == 0u ? man * uintBitsToFloat(0x3b000000u)
                              : (1.0 + man / 8.0) * uintBitsToFloat((e + 120u) << 23);
    return raw * 0.5;
}
