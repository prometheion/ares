//! Shader modules, the compute pipelines made from them, and launches.
//!
//! A kernel here is a pipeline with no descriptor sets: its inputs are the
//! push constants of the launch, which carry buffer addresses and sizes. That
//! is what lets a launch be "bind (if different), push, dispatch" with nothing
//! to allocate, write or update per launch.

use std::ffi::CString;
use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{Check, Error, Result};
use crate::instance::PUSH_CONSTANT_BYTES;
use crate::memory::DevicePtr;
use crate::stream::Stream;

struct ModuleInner {
    device: Device,
    raw: vk::ShaderModule,
    /// The entry points the SPIR-V declares.
    entries: Vec<String>,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        // SAFETY: pipelines made from the module keep it alive through their
        // own handle, and a shader module is not needed once they exist; it
        // is destroyed once.
        unsafe { self.device.raw().destroy_shader_module(self.raw, None) };
    }
}

/// Compiled kernels (SPIR-V) loaded on one device.
#[derive(Clone)]
pub struct Module(Arc<ModuleInner>);

impl std::fmt::Debug for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Module({:?} on {:?})", self.0.entries, self.0.device)
    }
}

const SPIRV_MAGIC: u32 = 0x0723_0203;

/// The 32-bit words of a SPIR-V binary, checked for size and magic number.
pub(crate) fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
        return Err(Error::invalid(
            "SPIR-V",
            format!("{} bytes are not a SPIR-V binary (a multiple of 4, at least 20)", bytes.len()),
        ));
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    let words: Vec<u32> = chunks.iter().map(|chunk| u32::from_le_bytes(*chunk)).collect();
    if words[0] != SPIRV_MAGIC {
        return Err(Error::invalid("SPIR-V", format!("bad magic number {:#x}", words[0])));
    }
    Ok(words)
}

/// The names of the entry points a SPIR-V binary declares (`OpEntryPoint`).
pub(crate) fn entry_points(words: &[u32]) -> Result<Vec<String>> {
    const OP_ENTRY_POINT: u32 = 15;
    let mut names = Vec::new();
    // The header is five words; instructions follow, each starting with its
    // word count in the high half and its opcode in the low half.
    let mut at = 5;
    while at < words.len() {
        let (count, opcode) = ((words[at] >> 16) as usize, words[at] & 0xffff);
        if count == 0 || at + count > words.len() {
            return Err(Error::invalid("SPIR-V", format!("a broken instruction at word {at}")));
        }
        if opcode == OP_ENTRY_POINT && count >= 4 {
            // Operands: execution model, entry point id, then the name as a
            // NUL-terminated string packed little-endian into words.
            let bytes: Vec<u8> =
                words[at + 3..at + count].iter().flat_map(|w| w.to_le_bytes()).collect();
            let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
            names.push(String::from_utf8_lossy(&bytes[..end]).into_owned());
        }
        at += count;
    }
    Ok(names)
}

impl Module {
    /// Loads a SPIR-V binary. Its kernels must follow the conventions of
    /// `shaders/common.glsl`: no descriptor sets, at most 128 bytes of push
    /// constants.
    pub fn load(device: &Device, spirv: &[u8]) -> Result<Self> {
        let words = spirv_words(spirv)?;
        let entries = entry_points(&words)?;
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: the device is alive and `words` is a SPIR-V binary of the
        // length it declares; the driver copies it.
        let raw = unsafe { device.raw().create_shader_module(&info, None) }
            .check("vkCreateShaderModule")?;
        Ok(Self(Arc::new(ModuleInner { device: device.clone(), raw, entries })))
    }

    /// The kernel with entry point `name`. The returned handle keeps the module
    /// loaded. Making the pipeline compiles the shader for the GPU, which takes
    /// milliseconds: do it at load, not per launch.
    pub fn function(&self, name: &str) -> Result<Function> {
        self.function_with(name, &[])
    }

    /// Like [`function`](Self::function), with values for the shader's
    /// specialization constants as `(constant id, value)`. Every combination
    /// is its own pipeline.
    pub fn function_with(&self, name: &str, constants: &[(u32, u32)]) -> Result<Function> {
        if !self.0.entries.iter().any(|entry| entry == name) {
            return Err(Error::NoSuchKernel(name.to_owned()));
        }
        let c_name = CString::new(name).map_err(|_| Error::NoSuchKernel(name.to_owned()))?;
        let entries: Vec<vk::SpecializationMapEntry> = constants
            .iter()
            .enumerate()
            .map(|(i, (id, _))| vk::SpecializationMapEntry {
                constant_id: *id,
                offset: (i * 4) as u32,
                size: 4,
            })
            .collect();
        let data: Vec<u8> = constants.iter().flat_map(|(_, value)| value.to_le_bytes()).collect();
        let specialization = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.0.raw)
            .name(&c_name)
            .specialization_info(&specialization);
        let info =
            vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.0.device.0.layout);
        let device = &self.0.device;
        // SAFETY: the module and layout are alive; the specialization data
        // outlives the call.
        let pipelines = unsafe {
            device.raw().create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, code)| Error::vk("vkCreateComputePipelines", code))?;
        Ok(Function(Arc::new(FunctionInner { module: self.clone(), raw: pipelines[0] })))
    }
}

struct FunctionInner {
    module: Module,
    raw: vk::Pipeline,
}

impl Drop for FunctionInner {
    fn drop(&mut self) {
        let device = &self.module.0.device;
        // Commands that bind the pipeline may still be recorded or running.
        let _ = device.synchronize();
        // SAFETY: nothing enqueued uses the pipeline any more; destroyed once.
        unsafe { device.raw().destroy_pipeline(self.raw, None) };
    }
}

/// A kernel that can be launched.
#[derive(Clone)]
pub struct Function(Arc<FunctionInner>);

impl std::fmt::Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Function({:?})", self.0.module)
    }
}

/// The most arguments a kernel of this engine takes: as many 8-byte slots as
/// fit the 128 bytes of push constants every Vulkan device guarantees.
pub const MAX_ARGS: usize = PUSH_CONSTANT_BYTES as usize / 8;

/// The arguments of one launch, in the kernel's push constant order.
///
/// Each value is placed at the next offset that is a multiple of its size, as
/// a `std430` push constant block lays out its members, so the builder methods
/// must be called with the types the kernel declares, in its order. An address
/// is 8 bytes, a `uint`, `int` or `float` 4.
#[derive(Clone, Copy, Debug)]
pub struct Args {
    bytes: [u8; PUSH_CONSTANT_BYTES as usize],
    len: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self { bytes: [0; PUSH_CONSTANT_BYTES as usize], len: 0 }
    }
}

impl Args {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(mut self, value: &[u8]) -> Self {
        let at = self.len.next_multiple_of(value.len());
        assert!(
            at + value.len() <= self.bytes.len(),
            "a kernel takes at most {} bytes of arguments",
            self.bytes.len()
        );
        self.bytes[at..at + value.len()].copy_from_slice(value);
        self.len = at + value.len();
        self
    }

    /// A buffer address (`buffer_reference` member).
    pub fn ptr(self, value: DevicePtr) -> Self {
        self.push(&value.0.to_le_bytes())
    }

    /// A `uint` member.
    pub fn u32(self, value: u32) -> Self {
        self.push(&value.to_le_bytes())
    }

    /// An `int` member.
    pub fn i32(self, value: i32) -> Self {
        self.push(&value.to_le_bytes())
    }

    /// A `uint64_t` member.
    pub fn u64(self, value: u64) -> Self {
        self.push(&value.to_le_bytes())
    }

    /// A `float` member.
    pub fn f32(self, value: f32) -> Self {
        self.push(&value.to_le_bytes())
    }

    /// The bytes to push: the arguments, padded to the 8 bytes the largest
    /// member of a block aligns to.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len.next_multiple_of(8)]
    }
}

impl Function {
    pub fn device(&self) -> &Device {
        &self.0.module.0.device
    }

    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.0.raw
    }

    /// Enqueues one launch of the kernel on `stream`: `grid` is the number of
    /// workgroups in each dimension (the workgroup size is part of the
    /// shader), `args` its push constants. A grid with a zero dimension does
    /// nothing.
    ///
    /// # Safety
    ///
    /// `args` must match the kernel's push constant block in number, order and
    /// type. Every address argument must address live memory that this kernel
    /// may access as it does, for as long as the launch has not run; the
    /// device does not check addresses, and a wrong one can hang or reset the
    /// GPU.
    pub unsafe fn launch(&self, stream: &Stream, grid: [u32; 3], args: Args) -> Result<()> {
        stream.dispatch(self, grid, &args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_are_laid_out_like_a_std430_block() {
        let args = Args::new().u32(7).ptr(DevicePtr(0x1122_3344_5566_7788)).u32(9).f32(1.0);
        let bytes = args.as_bytes();
        // The u32 first, then padding up to the address's alignment.
        assert_eq!(&bytes[0..4], &7u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &[0; 4]);
        assert_eq!(&bytes[8..16], &0x1122_3344_5566_7788u64.to_le_bytes());
        assert_eq!(&bytes[16..20], &9u32.to_le_bytes());
        assert_eq!(&bytes[20..24], &1.0f32.to_le_bytes());
        assert_eq!(bytes.len(), 24);
    }

    #[test]
    fn pushed_bytes_are_padded_to_eight() {
        let args = Args::new().ptr(DevicePtr::NULL).u32(1);
        assert_eq!(args.as_bytes().len(), 16);
        assert!(Args::new().as_bytes().is_empty());
    }

    #[test]
    #[should_panic(expected = "at most 128 bytes")]
    fn too_many_arguments_are_refused() {
        let mut args = Args::new();
        for _ in 0..=MAX_ARGS {
            args = args.u64(0);
        }
        let _ = args;
    }

    #[test]
    fn spirv_is_checked_and_its_entry_points_are_found() {
        assert!(spirv_words(&[]).is_err());
        assert!(spirv_words(&[0u8; 20]).is_err());
        assert!(spirv_words(&[1, 2, 3]).is_err());
        let image = include_bytes!(concat!(env!("OUT_DIR"), "/add_f32.spv"));
        let words = spirv_words(image).unwrap();
        assert_eq!(entry_points(&words).unwrap(), ["main"]);
        // An instruction that claims to run past the end is refused instead of
        // read past it, and so is one of length zero.
        let mut broken = words.clone();
        broken[5] = (0xffff << 16) | (broken[5] & 0xffff);
        assert!(entry_points(&broken).is_err());
        broken[5] &= 0xffff;
        assert!(entry_points(&broken).is_err());
    }
}
