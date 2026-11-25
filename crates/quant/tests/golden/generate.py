#!/usr/bin/env python3
"""Generate the golden fixtures of ares-quant from ggml's own C code.

The Rust crate is a port of ggml's quantization functions, so its tests must not
compare it with anything derived from the port. This script loads the built
`libggml-base.so` (the library only: no llama binary runs and no GPU is touched)
and produces, for every type ggml can convert to float, a small binary file with
input blocks and the f32 values ggml's C code decodes them to. For the encoders
it produces input floats and the bytes ggml's `quantize_row_*_ref` writes.

Rerun (the output is deterministic, so a rerun changes nothing unless the
library behaves differently):

    python3 -I crates/quant/tests/golden/generate.py <llama.cpp>

<llama.cpp> is a llama.cpp tree built with CMake into its `build/` directory: the
argument, else the LLAMA_CPP_DIR environment variable; there is no default. Build
it from a fresh clone, on the CPU (the library is all this needs):

    git clone https://github.com/ggml-org/llama.cpp $LLAMA_CPP_DIR
    cmake -S $LLAMA_CPP_DIR -B $LLAMA_CPP_DIR/build -DGGML_CUDA=OFF
    cmake --build $LLAMA_CPP_DIR/build --target ggml-base -j 6

The fixtures checked in came from the llama.cpp fork
(https://github.com/Piggidragon/llama.cpp) at f8f346786. Needs numpy.

Files (all little endian), written next to this script:

    dequant_<type>.bin   magic "EQGD", u32 version=1, u32 ggml type id,
                         u32 values per block, u32 bytes per block, u32 n_blocks,
                         u32 n_quantized, u32 reserved
                         then n_blocks * bytes_per_block block bytes
                         then n_blocks * values_per_block f32 reference values.
                         The first n_quantized blocks were made by ggml's
                         quantizer from float data; the rest are raw bytes.
    quantize_<type>.bin  magic "EQGQ", same header (n_blocks counts blocks, so for
                         f16 and bf16 it is the number of values; n_quantized is
                         unused and 0)
                         then n_blocks * values_per_block f32 inputs
                         then n_blocks * bytes_per_block reference bytes.

Where the reference values come from, per type:

  * to_float of `ggml_get_type_traits(type)` for every type that has one;
  * `dequantize_row_q8_K`, called by symbol, for Q8_K: ggml defines and exports
    it but does not register it in the type traits (Q8_K is an activation
    format);
  * F32: ggml has no to_float (nothing to convert), the reference is the input.
  * Q8_1 has no decoder in ggml at all, so it has no fixture and the crate
    reports it as unsupported.

Quantized blocks come from `ggml_quantize_chunk` (which is the `quantize_row_*_ref`
function for the types that have one). Types whose quantizer needs an importance
matrix (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS) get a fixed pseudo-random positive one.
Q8_K has no chunk quantizer; `quantize_row_q8_K_ref` is called by symbol.

Raw random blocks:

  * Every byte is pseudo-random, except the scale fields, which are rewritten
    with finite values: an f16 or f32 scale that is NaN or infinity would turn
    the whole block into NaN or infinity and make the comparison of the other
    fields meaningless. f16 scales are drawn from the whole finite range
    (zero, subnormals, tiny, ordinary, huge, both signs). MXFP4's E8M0 byte and
    NVFP4's UE4M3 bytes keep all 256 values (they have no infinity; the largest
    MXFP4 scales overflow f32 to infinity after multiplication, which is
    deterministic and compared bitwise like everything else). The script checks
    that no reference value is NaN, and none is infinite except for MXFP4.
  * For the codebook types, random indices would need hundreds of blocks to hit
    every table entry. The index fields of the raw blocks are instead filled
    from shuffled permutations of the whole index range, so that every entry of
    every table (grid and sign tables) is used at least once; the script checks
    this by reading the indices back out of the finished blocks. TQ1_0 gets the
    same treatment for its base-3 bytes, MXFP4 for the boundary values of its
    E8M0 exponent byte (0 and 1 are denormal scales) and NVFP4 for all 256
    values of its UE4M3 scale bytes. Everything else in those blocks (scales,
    delta bits, the other index fields) stays random.
"""

import ctypes
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
USAGE = "usage: generate.py <llama.cpp>, or LLAMA_CPP_DIR=<llama.cpp> (a built tree; see the head of this file)"
SEED = 20261109

# ggml_type ids that have a fixture (decoding).
DEQUANT_IDS = [
    0, 1, 30,  # f32 f16 bf16
    2, 3, 6, 7, 8,  # q4_0 q4_1 q5_0 q5_1 q8_0
    10, 11, 12, 13, 14, 15,  # q2_K .. q6_K q8_K
    16, 17, 18, 19, 20, 21, 22, 23, 29,  # iq2_xxs iq2_xs iq3_xxs iq1_s iq4_nl iq3_s iq2_s iq4_xs iq1_m
    34, 35,  # tq1_0 tq2_0
    39, 40, 41, 42,  # mxfp4 nvfp4 q1_0 q2_0
]
# ggml_type ids that have an encoder fixture.
QUANT_IDS = [1, 30, 2, 3, 6, 7, 8]
Q8_K = 15

# (blocks made by the quantizer, raw random blocks) per type. The codebook types
# need enough raw blocks to cover their tables: see COVERAGE below.
COUNTS = {
    "f32": (0, 512),
    "f16": (0, 1280),  # random patterns, on top of 64 * 16 structured ones
    "bf16": (0, 512),  # random patterns, on top of 512 * 4 structured ones
    "q4_0": (24, 40), "q4_1": (24, 40), "q5_0": (24, 40), "q5_1": (24, 40), "q8_0": (24, 40),
    "iq4_nl": (24, 40), "mxfp4": (24, 40),
    "nvfp4": (16, 64), "q1_0": (12, 20), "q2_0": (20, 36),
    "q2_K": (16, 16), "q3_K": (16, 16), "q4_K": (16, 16), "q5_K": (16, 16), "q6_K": (16, 16),
    "q8_K": (12, 12),
    "iq2_xxs": (16, 12), "iq2_xs": (16, 20), "iq2_s": (16, 34), "iq3_xxs": (16, 8),
    "iq3_s": (16, 10), "iq1_s": (16, 66), "iq1_m": (16, 66), "iq4_xs": (16, 16),
    "tq1_0": (16, 16), "tq2_0": (16, 16),
}
# Blocks (= rows of one block) per encoder fixture; the last 64 of the legacy ones are
# rounding-boundary rows. f16 and bf16 take whatever their input builders produce.
QUANT_COUNTS = {"q4_0": 160, "q4_1": 160, "q5_0": 160, "q5_1": 160, "q8_0": 160}

# Offsets of the f16 scale fields inside one block (see block_* in ggml-common.h).
F16_SCALES = {
    "q4_0": [0], "q4_1": [0, 2], "q5_0": [0], "q5_1": [0, 2], "q8_0": [0],
    "q2_K": [80, 82], "q3_K": [108], "q4_K": [0, 2], "q5_K": [0, 2], "q6_K": [208],
    "iq2_xxs": [0], "iq2_xs": [0], "iq2_s": [0], "iq3_xxs": [0], "iq3_s": [0],
    "iq1_s": [0], "iq4_nl": [0], "iq4_xs": [0], "tq1_0": [52], "tq2_0": [64],
    "q1_0": [0], "q2_0": [0],
}


class Traits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


ROW_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)


class Ggml:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        lib = self.lib
        lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
        lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
        ]
        lib.ggml_row_size.restype = ctypes.c_size_t
        lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.dequantize_row_q8_K.restype = None
        lib.dequantize_row_q8_K.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        lib.quantize_row_q8_K_ref.restype = None
        lib.quantize_row_q8_K_ref.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

    def traits(self, ty):
        return self.lib.ggml_get_type_traits(ty).contents

    def name(self, ty):
        return self.traits(ty).type_name.decode()

    def geometry(self, ty):
        t = self.traits(ty)
        return int(t.blck_size), int(t.type_size)

    def dequantize(self, ty, blocks):
        """The f32 values ggml decodes the (n, bytes) block array to."""
        values, size = self.geometry(ty)
        n = blocks.shape[0]
        blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
        assert blocks.shape[1] == size
        out = np.empty(n * values, dtype=np.float32)
        if ty == 0:
            return blocks.reshape(-1).view(np.float32).copy()
        if ty == Q8_K:
            self.lib.dequantize_row_q8_K(blocks.ctypes.data, out.ctypes.data, n * values)
            return out
        fn = self.traits(ty).to_float
        assert fn, f"{self.name(ty)} has no to_float"
        ROW_FN(fn)(blocks.ctypes.data, out.ctypes.data, n * values)
        return out

    def from_float_ref(self, ty, x):
        """The bytes of ggml's quantize_row_*_ref for the 1-D f32 array x."""
        values, size = self.geometry(ty)
        assert x.size % values == 0
        n = x.size // values
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.zeros((n, size), dtype=np.uint8)
        fn = self.traits(ty).from_float_ref
        assert fn, f"{self.name(ty)} has no from_float_ref"
        ROW_FN(fn)(x.ctypes.data, out.ctypes.data, x.size)
        return out

    def quantize(self, ty, rows, rng):
        """Blocks ggml's quantizer makes from a (n_rows, n_per_row) f32 array."""
        values, size = self.geometry(ty)
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        n_rows, n_per_row = rows.shape
        assert n_per_row % values == 0
        n_blocks = n_rows * n_per_row // values
        out = np.zeros((n_blocks, size), dtype=np.uint8)
        if ty == Q8_K:
            self.lib.quantize_row_q8_K_ref(rows.ctypes.data, out.ctypes.data, rows.size)
            return out
        imatrix = None
        if self.lib.ggml_quantize_requires_imatrix(ty):
            # Positive weights of varied size, one per column, shared by all rows.
            imatrix = np.ascontiguousarray(rng.uniform(0.05, 2.0, n_per_row), dtype=np.float32)
        written = self.lib.ggml_quantize_chunk(
            ty, rows.ctypes.data, out.ctypes.data, 0, n_rows, n_per_row,
            imatrix.ctypes.data if imatrix is not None else None,
        )
        assert written == n_blocks * size, (written, n_blocks * size)
        return out


# --------------------------------------------------------------------------- data


def data_rows(rng, n_rows, n, extreme=False):
    """n_rows float32 rows of n values in many shapes: magnitudes from 1e-5 to
    1e3, both signs and one sign only, zeros, constants, ramps, spikes, outliers
    and values on a coarse grid. Classes repeat cyclically so each is present
    as soon as n_rows reaches the number of classes. `extreme` adds rows of
    magnitude 1e20 and 1e-20, where the f16 scale of a block overflows or
    underflows; only the simple encoders are fed those (the K-quant search loops
    square the values and would overflow f32)."""

    def spike():
        x = rng.normal(0, 0.01, n)
        x[rng.integers(n)] = 100.0 * rng.choice([-1, 1])
        return x

    def laplace_outliers():
        x = rng.laplace(0, 0.1, n)
        x[rng.integers(n, size=max(1, n // 32))] *= 10
        return x

    def sparse():
        return rng.normal(0, 1, n) * (rng.random(n) < 0.1)

    makers = [
        lambda: rng.normal(0, 1, n),
        lambda: rng.normal(0, 0.02, n),
        lambda: rng.normal(0, 300.0, n),
        lambda: rng.normal(0, 1e-5, n),
        lambda: rng.uniform(-1, 1, n),
        lambda: rng.uniform(0, 1, n),
        lambda: -rng.uniform(0.5, 3, n),
        laplace_outliers,
        sparse,
        lambda: np.zeros(n),
        lambda: np.full(n, 0.37),
        lambda: np.full(n, -2.5),
        lambda: np.linspace(-1, 1, n),
        spike,
        lambda: rng.choice([-0.7, 0.7], n),
        lambda: np.round(rng.normal(0, 3, n)),
        lambda: rng.normal(0, 1, n) * 10.0 ** rng.integers(-3, 4, n),
    ]
    if extreme:
        makers += [lambda: rng.normal(0, 1e20, n), lambda: rng.normal(0, 1e-20, n)]
    rows = [np.asarray(makers[i % len(makers)](), dtype=np.float64) for i in range(n_rows)]
    return np.asarray(rows, dtype=np.float32).reshape(n_rows, n)


def tie_rows(rng, n, count, kind, bound):
    """Rows whose values sit exactly on rounding boundaries of the encoders.

    The block scale s is a power of two, so d and 1/d are exact and every x/d is
    a multiple of 0.5 or one float step off it. kind "sym" is Q4_0 (bound 8),
    Q5_0 (16) and Q8_0 (127): the first value is +-bound*s, which makes the scale
    s. kind "range" is Q4_1 (bound 15) and Q5_1 (31): the first two values are
    min and min + bound*s, which makes the scale s."""
    rows = []
    for _ in range(count):
        s = np.float32(2.0 ** rng.integers(-12, 12))
        offs = np.float32(rng.choice([0.0, 0.5, 0.5, 0.5, 0.25]))
        k = rng.integers(-bound, bound, n).astype(np.float32)
        x = (s * (k + offs)).astype(np.float32)
        # one float step above or below the boundary for a third of the entries each
        tweak = rng.integers(0, 3, n)
        x = np.where(tweak == 1, np.nextafter(x, np.float32(np.inf)), x)
        x = np.where(tweak == 2, np.nextafter(x, np.float32(-np.inf)), x).astype(np.float32)
        if kind == "sym":
            x = np.clip(x, -s * bound, s * bound)
            x[0] = s * bound * rng.choice([-1, 1])
        else:
            lo = np.float32(s * rng.integers(-bound, 1))
            x = np.clip(x, lo, lo + s * bound)
            x[0], x[1] = lo, lo + s * bound
        rows.append(x)
    return np.asarray(rows, dtype=np.float32)


# ------------------------------------------------------------------- raw blocks


def rand_f16_bits(rng, n):
    """n finite f16 bit patterns over the whole range."""
    kind = rng.integers(0, 10, n)
    sign = rng.integers(0, 2, n).astype(np.uint16) << 15
    man = rng.integers(0, 1024, n).astype(np.uint16)
    exp = np.where(kind < 1, 0, np.where(kind < 8, rng.integers(4, 20, n), rng.integers(1, 31, n)))
    return (sign | (exp.astype(np.uint16) << 10) | man).astype(np.uint16)


def put_u16(blocks, offset, values):
    blocks[:, offset] = values & 0xFF
    blocks[:, offset + 1] = values >> 8


def fix_scales(name, blocks, rng):
    """Replace the scale fields of random blocks with finite values."""
    n = blocks.shape[0]
    for off in F16_SCALES.get(name, []):
        put_u16(blocks, off, rand_f16_bits(rng, n))
    if name == "q8_K":
        # f32 scale: exponent 100..150 (about 1e-9 .. 1e6), any mantissa, either sign.
        bits = (rng.integers(0, 2, n).astype(np.uint32) << 31) | \
               (rng.integers(100, 151, n).astype(np.uint32) << 23) | \
               rng.integers(0, 1 << 23, n).astype(np.uint32)
        blocks[:, 0:4] = bits.view(np.uint8).reshape(n, 4)
    if name == "iq1_m":
        # The f16 scale is spread over the top nibbles of four u16 at offset 48.
        f16 = rand_f16_bits(rng, n).astype(np.uint32)
        for k in range(4):
            lo = blocks[:, 48 + 2 * k].astype(np.uint32)
            hi = blocks[:, 49 + 2 * k].astype(np.uint32)
            word = (lo | (hi << 8)) & 0x0FFF | (((f16 >> (4 * k)) & 0xF) << 12)
            blocks[:, 48 + 2 * k] = word & 0xFF
            blocks[:, 49 + 2 * k] = word >> 8


def le32(b, off):
    return int(b[off]) | int(b[off + 1]) << 8 | int(b[off + 2]) << 16 | int(b[off + 3]) << 24


def set_bits32(b, off, shift, width, value):
    word = le32(b, off)
    mask = ((1 << width) - 1) << shift
    word = (word & ~mask) | (value << shift)
    for i in range(4):
        b[off + i] = (word >> (8 * i)) & 0xFF


def set_bits16(b, off, shift, width, value):
    word = int(b[off]) | int(b[off + 1]) << 8
    mask = ((1 << width) - 1) << shift
    word = (word & ~mask) | (value << shift)
    b[off] = word & 0xFF
    b[off + 1] = word >> 8


def set_bits8(b, off, shift, width, value):
    mask = ((1 << width) - 1) << shift
    b[off] = (int(b[off]) & ~mask & 0xFF) | (value << shift)


# Each group is (table name, number of entries, slots per block, setter(block, slot, value)).
# The setters write the bit fields ggml-quants.c reads for index `slot`; the READERS
# below read them back the way the dequantize_row_* functions do.
def _iq2_xxs():
    return [
        ("iq2xxs_grid", 256, 32, lambda b, s, v: b.__setitem__(2 + 8 * (s // 4) + s % 4, v)),
        ("ksigns_iq2xs", 128, 32, lambda b, s, v: set_bits32(b, 2 + 8 * (s // 4) + 4, 7 * (s % 4), 7, v)),
    ]


def _iq2_xs():
    base = lambda s: 2 + 2 * s
    return [
        ("iq2xs_grid", 512, 32, lambda b, s, v: set_bits16(b, base(s), 0, 9, v)),
        ("ksigns_iq2xs", 128, 32, lambda b, s, v: set_bits16(b, base(s), 9, 7, v)),
    ]


def _iq2_s():
    return [
        ("iq2s_grid", 1024, 32, lambda b, s, v: (b.__setitem__(2 + s, v & 0xFF),
                                                 set_bits8(b, 66 + s // 4, 2 * (s % 4), 2, v >> 8))),
        ("sign byte", 256, 32, lambda b, s, v: b.__setitem__(2 + 32 + s, v)),
    ]


def _iq3_xxs():
    return [
        ("iq3xxs_grid", 256, 64, lambda b, s, v: b.__setitem__(2 + s, v)),
        ("ksigns_iq2xs", 128, 32, lambda b, s, v: set_bits32(b, 66 + 4 * (s // 4), 7 * (s % 4), 7, v)),
    ]


def _iq3_s():
    # slot s: ib32 = s // 8, l = (s % 8) // 2, k = s % 2; qs[8*ib32 + 2*l + k] == qs[s]
    return [
        ("iq3s_grid", 512, 64, lambda b, s, v: (b.__setitem__(2 + s, v & 0xFF),
                                                set_bits8(b, 66 + s // 8, (s % 8), 1, v >> 8))),
        ("sign byte", 256, 32, lambda b, s, v: b.__setitem__(74 + s, v)),
    ]


def _iq1_s():
    # slot s: ib = s // 4, l = s % 4
    return [
        ("iq1s_grid", 2048, 32, lambda b, s, v: (b.__setitem__(2 + s, v & 0xFF),
                                                 set_bits16(b, 34 + 2 * (s // 4), 3 * (s % 4), 3, v >> 8))),
    ]


def _iq1_m():
    # slot s: ib = s // 4, k = s % 4; index low byte qs[4*ib + k], high 3 bits in
    # qh[2*ib + k // 2] at bit 0 (k even) or bit 4 (k odd).
    return [
        ("iq1s_grid", 2048, 32, lambda b, s, v: (b.__setitem__(s, v & 0xFF),
                                                 set_bits8(b, 32 + 2 * (s // 4) + (s % 4) // 2,
                                                           4 * (s % 2), 3, v >> 8))),
    ]


# MXFP4's E8M0 scale is an exponent byte. Zero and one are denormal scales with their own code
# path, 2 is where the normal formula starts, 255 is the largest; the rest is a spread.
E8M0_SAMPLE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 63, 64, 100, 125, 126, 127, 128, 129, 130, 131,
               200, 240, 250, 251, 252, 253, 254, 255]


def _mxfp4():
    return [("e8m0 scale", E8M0_SAMPLE, 1, lambda b, s, v: b.__setitem__(0, v))]


def _nvfp4():
    # All 256 UE4M3 bytes, including the ones with the unused top bit set and the NaN code 0x7f.
    return [("ue4m3 scale", 256, 4, lambda b, s, v: b.__setitem__(s, v))]


def _tq1_0():
    # Bytes 0..32 of qs are decoded with all five digits: make them a permutation
    # of every byte value.
    return [("tq1_0 byte", 256, 32, lambda b, s, v: b.__setitem__(s, v))]


COVERAGE = {
    "iq2_xxs": _iq2_xxs(), "iq2_xs": _iq2_xs(), "iq2_s": _iq2_s(), "iq3_xxs": _iq3_xxs(),
    "iq3_s": _iq3_s(), "iq1_s": _iq1_s(), "iq1_m": _iq1_m(), "tq1_0": _tq1_0(),
    "mxfp4": _mxfp4(), "nvfp4": _nvfp4(),
}


def domain(entries):
    """The values a coverage group must reach: 0..entries, or an explicit list."""
    return list(range(entries)) if isinstance(entries, int) else list(entries)


def read_indices(name, b):
    """The table indices a finished block uses, read like dequantize_row_* does.
    Returns {table: set}."""
    out = {}

    def add(table, v):
        out.setdefault(table, set()).add(int(v))

    if name == "iq2_xxs":
        for ib32 in range(8):
            base = 2 + 8 * ib32
            for l in range(4):
                add("iq2xxs_grid", b[base + l])
            aux1 = le32(b, base + 4)
            for l in range(4):
                add("ksigns_iq2xs", (aux1 >> (7 * l)) & 127)
    elif name == "iq2_xs":
        for i in range(32):
            q = int(b[2 + 2 * i]) | int(b[3 + 2 * i]) << 8
            add("iq2xs_grid", q & 511)
            add("ksigns_iq2xs", q >> 9)
    elif name == "iq2_s":
        qs, qh = b[2:66], b[66:74]
        for ib32 in range(8):
            for l in range(4):
                add("iq2s_grid", int(qs[4 * ib32 + l]) | ((int(qh[ib32]) << (8 - 2 * l)) & 0x300))
                add("sign byte", qs[32 + 4 * ib32 + l])
    elif name == "iq3_xxs":
        qs = b[2:66]
        for ib32 in range(8):
            aux = le32(b, 66 + 4 * ib32)
            for l in range(4):
                add("ksigns_iq2xs", (aux >> (7 * l)) & 127)
                add("iq3xxs_grid", qs[8 * ib32 + 2 * l])
                add("iq3xxs_grid", qs[8 * ib32 + 2 * l + 1])
    elif name == "iq3_s":
        qs, qh, signs = b[2:66], b[66:74], b[74:106]
        for ib32 in range(8):
            for l in range(4):
                add("iq3s_grid", int(qs[8 * ib32 + 2 * l]) | ((int(qh[ib32]) << (8 - 2 * l)) & 256))
                add("iq3s_grid", int(qs[8 * ib32 + 2 * l + 1]) | ((int(qh[ib32]) << (7 - 2 * l)) & 256))
                add("sign byte", signs[4 * ib32 + l])
    elif name == "iq1_s":
        qs = b[2:34]
        for ib in range(8):
            qh = int(b[34 + 2 * ib]) | int(b[35 + 2 * ib]) << 8
            for l in range(4):
                add("iq1s_grid", int(qs[4 * ib + l]) | (((qh >> (3 * l)) & 7) << 8))
    elif name == "iq1_m":
        qs, qh = b[0:32], b[32:48]
        for ib in range(8):
            q0, q1 = int(qh[2 * ib]), int(qh[2 * ib + 1])
            add("iq1s_grid", int(qs[4 * ib]) | ((q0 << 8) & 0x700))
            add("iq1s_grid", int(qs[4 * ib + 1]) | ((q0 << 4) & 0x700))
            add("iq1s_grid", int(qs[4 * ib + 2]) | ((q1 << 8) & 0x700))
            add("iq1s_grid", int(qs[4 * ib + 3]) | ((q1 << 4) & 0x700))
    elif name == "tq1_0":
        for i in range(32):
            add("tq1_0 byte", b[i])
    elif name == "mxfp4":
        add("e8m0 scale", b[0])
    elif name == "nvfp4":
        for i in range(4):
            add("ue4m3 scale", b[i])
    return out


def apply_coverage(name, blocks, rng):
    groups = COVERAGE.get(name)
    if not groups:
        return
    for table, entries, slots, setter in groups:
        values = np.array(domain(entries))
        total = blocks.shape[0] * slots
        reps = -(-total // len(values))
        seq = np.concatenate([rng.permutation(values) for _ in range(reps)])[:total]
        k = 0
        for b in blocks:
            for s in range(slots):
                setter(b, s, int(seq[k]))
                k += 1


def check_coverage(name, blocks):
    """Every entry of every table the type reads must be used by the raw blocks."""
    groups = COVERAGE.get(name)
    if not groups:
        return
    used = {}
    for b in blocks:
        for table, values in read_indices(name, b).items():
            used.setdefault(table, set()).update(values)
    for table, entries, _, _ in groups:
        wanted = set(domain(entries))
        missing = wanted - used.get(table, set())
        assert not missing, f"{name}: {len(missing)} of {len(wanted)} entries of {table} unused, e.g. {sorted(missing)[:5]}"


def random_blocks(name, n, size, rng):
    blocks = rng.integers(0, 256, (n, size), dtype=np.uint8)
    apply_coverage(name, blocks, rng)
    fix_scales(name, blocks, rng)
    check_coverage(name, blocks)
    return blocks


# ----------------------------------------------------------------------- output


def write_fixture(path, magic, ty, values, size, n_blocks, n_quantized, first, second):
    header = magic + struct.pack("<IIIIIII", 1, ty, values, size, n_blocks, n_quantized, 0)
    with open(path, "wb") as f:
        f.write(header)
        f.write(first.tobytes())
        f.write(second.tobytes())
    return os.path.getsize(path)


def dequant_fixture(g, ty, rng):
    name = g.name(ty)
    values, size = g.geometry(ty)
    n_quant, n_raw = COUNTS[name]
    parts = []
    if ty in (1, 30):
        # f16 / bf16: every (sign, exponent) pair with a spread of mantissas,
        # then random patterns. NaNs and infinities are included: their widening
        # is defined and compared bitwise.
        sign_exp = np.arange(64, dtype=np.uint32) << 10
        mans = np.array([0, 1, 2, 3, 4, 0x80, 0x100, 0x155, 0x200, 0x2AA, 0x300, 0x3C0, 0x3FC, 0x3FD, 0x3FE, 0x3FF])
        grid16 = (sign_exp[:, None] | mans[None, :]).reshape(-1).astype(np.uint16)
        if ty == 30:
            # bf16: 8 exponent bits + sign = 512 combinations, 7 mantissa bits.
            sign_exp = np.arange(512, dtype=np.uint32) << 7
            mans = np.array([0, 1, 0x40, 0x7F])
            grid16 = (sign_exp[:, None] | mans[None, :]).reshape(-1).astype(np.uint16)
        rand = rng.integers(0, 1 << 16, n_raw, dtype=np.uint32).astype(np.uint16)
        pats = np.concatenate([grid16, rand])
        raw = pats.view(np.uint8).reshape(-1, 2)
        blocks = raw
        n_quant = 0
    elif ty == 0:
        bits = rng.integers(0, 1 << 32, n_raw, dtype=np.uint64).astype(np.uint32)
        special = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001,
                            0xFFFFFFFF, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF], dtype=np.uint32)
        bits[: len(special)] = special
        blocks = bits.view(np.uint8).reshape(-1, 4)
        n_quant = 0
    else:
        rows = data_rows(rng, n_quant, values)
        quantized = g.quantize(ty, rows, rng)
        raw = random_blocks(name, n_raw, size, rng)
        blocks = np.concatenate([quantized, raw])
    expected = g.dequantize(ty, blocks)
    if name not in ("f32", "f16", "bf16"):  # these three get NaN and infinity patterns on purpose
        assert not np.isnan(expected).any(), f"{name}: NaN in the reference output"
        if name != "mxfp4":
            assert np.isfinite(expected).all(), f"{name}: infinity in the reference output"
    path = os.path.join(HERE, f"dequant_{name}.bin")
    n_blocks = blocks.shape[0]
    return name, write_fixture(path, b"EQGD", ty, values, size, n_blocks, n_quant, blocks, expected)


def f16_inputs(rng):
    """f32 inputs that stress float -> f16 rounding."""
    f = lambda bits: np.asarray(bits, dtype=np.uint32).view(np.float32)
    all16 = np.arange(0, 0x7C00, dtype=np.uint32)  # finite non-negative f16 patterns
    exact = f16_to_f32_np(all16)
    # Midpoints of neighbouring f16 values (exactly representable in f32) and one
    # f32 step either side, for the subnormals and the first normal binade in
    # full and for every 13th pair elsewhere.
    pairs = np.concatenate([np.arange(0, 0x0400), np.arange(0x0400, 0x0800, 8), np.arange(0x0800, 0x7BFF, 97)])
    mid = (exact[pairs] + exact[pairs + 1]) / np.float32(2)
    mid_bits = mid.view(np.uint32)
    near = np.concatenate([mid_bits, mid_bits - 1, mid_bits + 1])
    sample = exact[::127]
    specials = np.array([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
                         0x7F800001, 0xFFFFFFFF, 0x477FEFFF, 0x477FF000, 0x477FF001, 0x47800000,
                         0x7F7FFFFF, 0x33000000, 0x33000001, 0x32FFFFFF, 0x33800000, 0x38800000,
                         0x387FFFFF, 0x3F800000], dtype=np.uint32)
    rand = rng.integers(0, 1 << 32, 300, dtype=np.uint64).astype(np.uint32)
    # random magnitudes across the exponent range (random bits are mostly huge or tiny)
    scaled = (rng.normal(0, 1, 300) * 10.0 ** rng.uniform(-9, 5, 300)).astype(np.float32).view(np.uint32)
    bits = np.concatenate([near, sample.view(np.uint32), specials, rand, scaled])
    # both signs
    neg = bits[::4] | np.uint32(0x80000000)
    return f(np.concatenate([bits, neg]))


def f16_to_f32_np(bits):
    """f16 -> f32 for finite patterns, independent of ggml and of the crate."""
    bits = np.asarray(bits, dtype=np.uint32)
    exp = (bits >> 10) & 0x1F
    man = bits & 0x3FF
    val = np.where(exp == 0, man * 2.0 ** -24, (1024 + man) * 2.0 ** (exp.astype(np.int64) - 25.0))
    return val.astype(np.float32)


def bf16_inputs(rng):
    specials = np.array([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001,
                         0xFF80FFFF, 0x7F7FFFFF, 0x7F7F8000, 0x7F7F7FFF, 0x3F808000, 0x3F818000,
                         0x3F800001, 0x3F80FFFF, 0x00008000, 0x00018000, 0x00800000],
                        dtype=np.uint32)
    # ties and neighbours: every 16-bit prefix with low half 0x7FFF, 0x8000, 0x8001
    prefixes = (np.arange(0, 1 << 16, 61, dtype=np.uint32) << 16)
    ties = np.concatenate([prefixes | 0x7FFF, prefixes | 0x8000, prefixes | 0x8001])
    rand = rng.integers(0, 1 << 32, 600, dtype=np.uint64).astype(np.uint32)
    bits = np.concatenate([specials, ties, rand])
    return bits.view(np.float32)


def quant_fixture(g, ty, rng):
    name = g.name(ty)
    values, size = g.geometry(ty)
    if ty == 1:
        x = f16_inputs(rng)
    elif ty == 30:
        x = bf16_inputs(rng)
    else:
        count = QUANT_COUNTS[name]
        kind, bound = {2: ("sym", 8), 6: ("sym", 16), 8: ("sym", 127),
                       3: ("range", 15), 7: ("range", 31)}[ty]
        rows = data_rows(rng, count - 64, values, extreme=True)
        ties = tie_rows(rng, values, 64, kind, bound)
        x = np.concatenate([rows, ties]).reshape(-1)
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert x.size % values == 0
    assert np.isfinite(x).all() or ty in (1, 30)
    expected = g.from_float_ref(ty, x)
    path = os.path.join(HERE, f"quantize_{name}.bin")
    n_blocks = x.size // values
    return name, write_fixture(path, b"EQGQ", ty, values, size, n_blocks, 0, x, expected)


def llama_cpp_tree():
    """The llama.cpp tree of the argument, else of LLAMA_CPP_DIR."""
    if len(sys.argv) > 2:
        sys.exit(USAGE)
    tree = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLAMA_CPP_DIR", "")
    if not tree:
        sys.exit(USAGE)
    return os.path.abspath(tree)


def main():
    lib_path = os.path.join(llama_cpp_tree(), "build", "bin", "libggml-base.so")
    if not os.path.isfile(lib_path):
        sys.exit(f"{lib_path} is not there: build the tree first (see the head of this file)")
    g = Ggml(lib_path)
    total = 0
    for ty in DEQUANT_IDS:
        rng = np.random.default_rng([SEED, ty, 1])
        name, size = dequant_fixture(g, ty, rng)
        total += size
        print(f"dequant_{name}.bin  {size:>8} bytes")
    for ty in QUANT_IDS:
        rng = np.random.default_rng([SEED, ty, 2])
        name, size = quant_fixture(g, ty, rng)
        total += size
        print(f"quantize_{name}.bin {size:>8} bytes")
    print(f"total {total} bytes ({total / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
