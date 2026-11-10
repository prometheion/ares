//! Tensor element types as GGUF files store them.
//!
//! The numbering and the block geometry follow ggml (`enum ggml_type` and its
//! type traits), because GGUF is the weight format of the engine. A quantized
//! type stores values in fixed blocks: `block_values()` values take
//! `block_bytes()` bytes, and a tensor row always holds a whole number of
//! blocks.

use std::fmt;

/// Element type of a tensor in a GGUF file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

/// Every type the engine knows, in file-id order.
pub const ALL_TYPES: [GgmlType; 35] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q8_1,
    GgmlType::Q2_K,
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_K,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
    GgmlType::IQ3_XXS,
    GgmlType::IQ1_S,
    GgmlType::IQ4_NL,
    GgmlType::IQ3_S,
    GgmlType::IQ2_S,
    GgmlType::IQ4_XS,
    GgmlType::I8,
    GgmlType::I16,
    GgmlType::I32,
    GgmlType::I64,
    GgmlType::F64,
    GgmlType::IQ1_M,
    GgmlType::BF16,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
    GgmlType::MXFP4,
    GgmlType::NVFP4,
    GgmlType::Q1_0,
    GgmlType::Q2_0,
];

impl GgmlType {
    /// The type for a GGUF type id, or `None` for an id the engine does not know
    /// (including the ids ggml has retired).
    pub fn from_id(id: u32) -> Option<Self> {
        ALL_TYPES.iter().copied().find(|t| *t as u32 == id)
    }

    /// The id stored in GGUF files.
    pub fn id(self) -> u32 {
        self as u32
    }

    /// `(values per block, bytes per block)`.
    pub const fn block(self) -> (u32, u32) {
        use GgmlType::*;
        match self {
            F32 => (1, 4),
            F16 => (1, 2),
            BF16 => (1, 2),
            F64 => (1, 8),
            I8 => (1, 1),
            I16 => (1, 2),
            I32 => (1, 4),
            I64 => (1, 8),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            Q2_K => (256, 84),
            Q3_K => (256, 110),
            Q4_K => (256, 144),
            Q5_K => (256, 176),
            Q6_K => (256, 210),
            Q8_K => (256, 292),
            IQ2_XXS => (256, 66),
            IQ2_XS => (256, 74),
            IQ2_S => (256, 82),
            IQ3_XXS => (256, 98),
            IQ3_S => (256, 110),
            IQ1_S => (256, 50),
            IQ1_M => (256, 56),
            IQ4_NL => (32, 18),
            IQ4_XS => (256, 136),
            TQ1_0 => (256, 54),
            TQ2_0 => (256, 66),
            MXFP4 => (32, 17),
            NVFP4 => (64, 36),
            Q1_0 => (128, 18),
            Q2_0 => (64, 18),
        }
    }

    /// Values stored in one block (1 for plain types).
    pub const fn block_values(self) -> u32 {
        self.block().0
    }

    /// Bytes of one block.
    pub const fn block_bytes(self) -> u32 {
        self.block().1
    }

    /// Whether values are stored in quantized blocks.
    pub const fn is_quantized(self) -> bool {
        self.block_values() > 1
    }

    /// Bytes that `n_values` values take, or `None` when `n_values` is not a
    /// whole number of blocks or the size does not fit in `u64`.
    pub fn bytes_for(self, n_values: u64) -> Option<u64> {
        let (values, bytes) = self.block();
        if !n_values.is_multiple_of(u64::from(values)) {
            return None;
        }
        (n_values / u64::from(values)).checked_mul(u64::from(bytes))
    }

    /// Average bits one value takes.
    pub fn bits_per_value(self) -> f64 {
        let (values, bytes) = self.block();
        f64::from(bytes) * 8.0 / f64::from(values)
    }

    /// The name ggml uses, lower case (`"q4_K"`, `"f16"`, ...).
    pub const fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            BF16 => "bf16",
            F64 => "f64",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2_K => "q2_K",
            Q3_K => "q3_K",
            Q4_K => "q4_K",
            Q5_K => "q5_K",
            Q6_K => "q6_K",
            Q8_K => "q8_K",
            IQ2_XXS => "iq2_xxs",
            IQ2_XS => "iq2_xs",
            IQ2_S => "iq2_s",
            IQ3_XXS => "iq3_xxs",
            IQ3_S => "iq3_s",
            IQ1_S => "iq1_s",
            IQ1_M => "iq1_m",
            IQ4_NL => "iq4_nl",
            IQ4_XS => "iq4_xs",
            TQ1_0 => "tq1_0",
            TQ2_0 => "tq2_0",
            MXFP4 => "mxfp4",
            NVFP4 => "nvfp4",
            Q1_0 => "q1_0",
            Q2_0 => "q2_0",
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `id:values/bytes` for every type, as printed from gguf-py
    /// (`GGML_QUANT_SIZES`) of llama.cpp at the time of writing.
    const GGUF_PY: &str = "0:1/4 1:1/2 2:32/18 3:32/20 6:32/22 7:32/24 8:32/34 9:32/36 \
        10:256/84 11:256/110 12:256/144 13:256/176 14:256/210 15:256/292 16:256/66 \
        17:256/74 18:256/98 19:256/50 20:32/18 21:256/110 22:256/82 23:256/136 24:1/1 \
        25:1/2 26:1/4 27:1/8 28:1/8 29:256/56 30:1/2 34:256/54 35:256/66 39:32/17 \
        40:64/36 41:128/18 42:64/18";

    #[test]
    fn block_geometry_matches_gguf_py() {
        let mut seen = 0;
        for entry in GGUF_PY.split_whitespace() {
            let (id, geometry) = entry.split_once(':').unwrap();
            let (values, bytes) = geometry.split_once('/').unwrap();
            let ty = GgmlType::from_id(id.parse().unwrap())
                .unwrap_or_else(|| panic!("type id {id} is missing"));
            assert_eq!(ty.block_values(), values.parse::<u32>().unwrap(), "{ty}");
            assert_eq!(ty.block_bytes(), bytes.parse::<u32>().unwrap(), "{ty}");
            seen += 1;
        }
        assert_eq!(seen, ALL_TYPES.len());
    }

    #[test]
    fn ids_round_trip_and_retired_ids_are_unknown() {
        for ty in ALL_TYPES {
            assert_eq!(GgmlType::from_id(ty.id()), Some(ty));
        }
        for retired in [4, 5, 31, 32, 33, 36, 37, 38, 43, u32::MAX] {
            assert_eq!(GgmlType::from_id(retired), None);
        }
    }

    #[test]
    fn bytes_for_needs_whole_blocks() {
        assert_eq!(GgmlType::Q4_K.bytes_for(512), Some(288));
        assert_eq!(GgmlType::Q4_K.bytes_for(300), None);
        assert_eq!(GgmlType::F16.bytes_for(3), Some(6));
        assert_eq!(GgmlType::F64.bytes_for(u64::MAX), None);
    }
}
