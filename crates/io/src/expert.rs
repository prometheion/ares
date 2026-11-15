//! Where each expert of a mixture-of-experts model is in the files.
//!
//! A layer's experts are stored stacked: one tensor per matrix (gate, up,
//! down), three-dimensional with the expert on the last axis, so expert `e`
//! of a tensor is the byte range `offset + e * stride` of length `stride`.
//! The matrices of an expert are contiguous within their tensor and not
//! across tensors, so reading an expert means one read per tensor.

use ares_gguf::{Gguf, TensorInfo};

use crate::error::LayoutError;
use crate::file::FileId;

/// One contiguous range of one file: a read to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRange {
    /// The part of the model the range is in; the same number as the
    /// [`FileId`] of a reader opened with
    /// [`Reader::open_gguf`](crate::Reader::open_gguf).
    pub file: FileId,
    pub offset: u64,
    pub len: usize,
}

#[derive(Debug)]
struct Stacked {
    file: FileId,
    /// File offset of expert 0.
    base: u64,
    /// Bytes from one expert to the next in this tensor.
    stride: usize,
    /// Bytes of one expert's range: the stride, unless the tensor holds
    /// something else of every expert as well ([`ExpertSpan`]).
    len: usize,
}

/// A matrix of every expert of a layer, given by where it is in a file rather
/// than by a tensor of the model: expert `e` has `len` bytes at
/// `base + e * stride`. For a file that keeps two matrices of an expert in one
/// tensor, and for experts in a file of their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSpan {
    /// The file, as the reader that serves the map numbers it.
    pub file: FileId,
    pub base: u64,
    pub stride: usize,
    pub len: usize,
}

#[derive(Debug)]
struct LayerExperts {
    n_experts: usize,
    tensors: Vec<Stacked>,
}

/// The byte ranges of every expert of every MoE layer of a model.
///
/// It copies what it needs out of the opened [`Gguf`] and keeps no reference
/// to it. Layers without experts (dense layers, say) have no entry.
#[derive(Debug)]
pub struct ExpertMap {
    layers: Vec<Option<LayerExperts>>,
}

impl ExpertMap {
    /// Builds the map from the tensor names of each layer: `layers[l]` lists
    /// the stacked tensors of layer `l` (an empty list means the layer has
    /// no experts). The order of the names is the order of the ranges
    /// [`ExpertMap::ranges`] yields.
    ///
    /// # Errors
    ///
    /// A name that is not in the model, a tensor that is not 3-dimensional or
    /// does not divide into equal experts, and tensors of one layer that hold
    /// different numbers of experts.
    pub fn new<S: AsRef<str>>(gguf: &Gguf, layers: &[Vec<S>]) -> Result<Self, LayoutError> {
        let layers = layers
            .iter()
            .enumerate()
            .map(|(layer, names)| {
                if names.is_empty() {
                    return Ok(None);
                }
                let infos = names
                    .iter()
                    .map(|name| {
                        gguf.tensor(name.as_ref())
                            .ok_or_else(|| LayoutError::UnknownTensor(name.as_ref().to_owned()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                layer_of(layer, &infos).map(Some)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { layers })
    }

    /// Finds the stacked expert tensors of a model by name: the 3-dimensional
    /// tensors named `blk.<layer>.<something containing a pattern>`. With the
    /// patterns `["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]` that is
    /// the gate, up and down matrices of every MoE layer of a llama.cpp
    /// style model. A layer must have a tensor for each pattern or for none.
    pub fn discover(gguf: &Gguf, patterns: &[&str]) -> Result<Self, LayoutError> {
        let mut layers: Vec<Vec<Option<&TensorInfo>>> = Vec::new();
        for info in gguf.tensors() {
            if info.dims.as_slice().len() != 3 {
                continue;
            }
            let Some(layer) = block_index(&info.name) else { continue };
            for (slot, pattern) in patterns.iter().enumerate() {
                if !info.name.contains(pattern) {
                    continue;
                }
                if layers.len() <= layer {
                    layers.resize_with(layer + 1, Vec::new);
                }
                let tensors = &mut layers[layer];
                tensors.resize(patterns.len(), None);
                if let Some(first) = tensors[slot] {
                    return Err(LayoutError::AmbiguousMatch {
                        layer,
                        pattern: (*pattern).to_owned(),
                        first: first.name.clone(),
                        second: info.name.clone(),
                    });
                }
                tensors[slot] = Some(info);
            }
        }
        if layers.iter().all(Vec::is_empty) {
            return Err(LayoutError::NoMatch {
                patterns: patterns.iter().map(|p| (*p).to_owned()).collect(),
            });
        }
        let layers = layers
            .into_iter()
            .enumerate()
            .map(|(layer, slots)| {
                if slots.is_empty() {
                    return Ok(None);
                }
                let mut infos = Vec::with_capacity(slots.len());
                for (slot, info) in slots.into_iter().enumerate() {
                    infos.push(info.ok_or_else(|| LayoutError::IncompleteLayer {
                        layer,
                        missing: patterns[slot].to_owned(),
                    })?);
                }
                layer_of(layer, &infos).map(Some)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { layers })
    }

    /// Puts a layer in front of the others: it becomes layer 0 and every other
    /// layer's number grows by one. The layer has `n_experts` experts whose
    /// matrices are `spans`, in the order [`ExpertMap::ranges`] yields them.
    pub fn insert_first(&mut self, n_experts: usize, spans: &[ExpertSpan]) {
        let tensors = spans
            .iter()
            .map(|s| Stacked { file: s.file, base: s.base, stride: s.stride, len: s.len })
            .collect();
        self.layers.insert(0, Some(LayerExperts { n_experts, tensors }));
    }

    /// One more than the number of the last layer with experts.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// The layers that have experts.
    pub fn moe_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.iter().enumerate().filter(|(_, l)| l.is_some()).map(|(i, _)| i)
    }

    /// How many experts `layer` has, or `None` for a layer without experts.
    pub fn n_experts(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer)?.as_ref().map(|l| l.n_experts)
    }

    /// The bytes of one expert of `layer` over all its tensors; the same for
    /// every expert of the layer.
    pub fn expert_bytes(&self, layer: usize) -> Option<u64> {
        let layer = self.layers.get(layer)?.as_ref()?;
        Some(layer.tensors.iter().map(|t| t.len as u64).sum())
    }

    /// The ranges to read for expert `expert` of `layer`, one per tensor, in
    /// the order the tensors were named. `None` when the layer has no
    /// experts or `expert` is out of range.
    pub fn ranges(
        &self,
        layer: usize,
        expert: usize,
    ) -> Option<impl DoubleEndedIterator<Item = ExpertRange> + ExactSizeIterator + Clone + '_> {
        let layer = self.layers.get(layer)?.as_ref()?;
        if expert >= layer.n_experts {
            return None;
        }
        Some(layer.tensors.iter().map(move |t| ExpertRange {
            file: t.file,
            offset: t.base + (expert * t.stride) as u64,
            len: t.len,
        }))
    }
}

/// `blk.12.ffn_up_exps.weight` -> 12.
fn block_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (digits, _) = rest.split_once('.')?;
    digits.parse().ok()
}

fn layer_of(layer: usize, infos: &[&TensorInfo]) -> Result<LayerExperts, LayoutError> {
    let first = infos[0];
    let n_experts = first.dims.get(2);
    let tensors = infos
        .iter()
        .map(|info| {
            if info.dims.as_slice().len() != 3 {
                return Err(LayoutError::NotStacked {
                    name: info.name.clone(),
                    dims: info.dims.as_slice().len(),
                });
            }
            let experts = info.dims.get(2);
            if experts != n_experts {
                return Err(LayoutError::ExpertCountMismatch {
                    layer,
                    name: info.name.clone(),
                    experts,
                    first: first.name.clone(),
                    first_experts: n_experts,
                });
            }
            let indivisible =
                || LayoutError::Indivisible { name: info.name.clone(), size: info.size, experts };
            if experts == 0 || info.size % experts != 0 {
                return Err(indivisible());
            }
            let stride = usize::try_from(info.size / experts).map_err(|_| indivisible())?;
            Ok(Stacked { file: FileId(info.part), base: info.offset, stride, len: stride })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LayerExperts { n_experts: n_experts as usize, tensors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_put_in_front_is_layer_0_and_reads_the_ranges_it_was_given() {
        let mut map = ExpertMap { layers: vec![None] };
        // Two matrices in one tensor of 100 bytes an expert, and one of 40.
        map.insert_first(
            3,
            &[
                ExpertSpan { file: FileId(2), base: 1000, stride: 100, len: 60 },
                ExpertSpan { file: FileId(2), base: 1060, stride: 100, len: 40 },
                ExpertSpan { file: FileId(2), base: 5000, stride: 40, len: 40 },
            ],
        );
        assert_eq!((map.n_layers(), map.n_experts(0), map.n_experts(1)), (2, Some(3), None));
        assert_eq!(map.expert_bytes(0), Some(140));
        let ranges: Vec<ExpertRange> = map.ranges(0, 2).expect("an expert").collect();
        assert_eq!(
            ranges,
            [
                ExpertRange { file: FileId(2), offset: 1200, len: 60 },
                ExpertRange { file: FileId(2), offset: 1260, len: 40 },
                ExpertRange { file: FileId(2), offset: 5080, len: 40 },
            ]
        );
        assert!(map.ranges(0, 3).is_none());
    }

    #[test]
    fn block_numbers_come_from_the_name() {
        assert_eq!(block_index("blk.0.ffn_up_exps.weight"), Some(0));
        assert_eq!(block_index("blk.127.ffn_up_exps.weight"), Some(127));
        assert_eq!(block_index("token_embd.weight"), None);
        assert_eq!(block_index("blk.x.ffn_up_exps.weight"), None);
        assert_eq!(block_index("blk.3"), None);
    }
}
