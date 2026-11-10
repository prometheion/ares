//! The synchronous path between host memory and device buffers, behind
//! [`DeviceBuffer::write`](crate::DeviceBuffer::write) and
//! [`read`](crate::DeviceBuffer::read).
//!
//! Ordinary memory cannot be the source of a DMA copy, so data goes through a
//! staging buffer of the device, in chunks. Two chunk slots alternate: while
//! the device copies one, the CPU fills (or empties) the other, which keeps a
//! large upload close to the slower of memcpy and the bus instead of the sum of
//! both. Every chunk is its own submission with its own fence, and the call
//! returns only after the last fence has signalled, so "returned" means "the
//! data is there", not "the driver has the data".

use ash::vk;

use crate::device::{Device, lock};
use crate::error::{Check, Result};
use crate::memory::HostMemory;

/// Bytes per chunk. Large enough that a chunk's submission cost is lost in
/// its copy time (8 MiB take about 0.4 ms on PCIe 4.0 x16), small enough
/// that the two slots are modest host memory (16 MiB per device).
const CHUNK: usize = 8 << 20;

/// A global memory barrier for the commands of this module and the streams:
/// every earlier write by a kernel or a copy is visible to the later commands.
pub(crate) fn full_barrier() -> vk::MemoryBarrier<'static> {
    vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::TRANSFER_WRITE,
        )
}

/// The stages that run the engine's commands.
pub(crate) const WORK_STAGES: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
    vk::PipelineStageFlags::COMPUTE_SHADER.as_raw() | vk::PipelineStageFlags::TRANSFER.as_raw(),
);

enum Slot {
    Free,
    /// Submitted; for a read, the range of the destination it fills.
    Busy(Option<(usize, usize)>),
}

pub(crate) struct Transfer {
    pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; 2],
    fences: [vk::Fence; 2],
    slots: [Slot; 2],
    staging: Option<HostMemory>,
}

/// One piece of a transfer: `len` bytes between a slot of the staging memory
/// and `buffer_offset` of the device buffer.
struct Chunk {
    slot: usize,
    buffer_offset: u64,
    len: usize,
    is_write: bool,
    /// Where the piece starts in the host's data.
    host_start: usize,
}

/// The host side of a transfer.
enum Host<'a> {
    Source(&'a [u8]),
    Sink(&'a mut [u8]),
}

impl Host<'_> {
    fn len(&self) -> usize {
        match self {
            Host::Source(data) => data.len(),
            Host::Sink(data) => data.len(),
        }
    }
}

impl Transfer {
    fn create(device: &Device) -> Result<Self> {
        let mut transfer = Self {
            pool: vk::CommandPool::null(),
            command_buffers: [vk::CommandBuffer::null(); 2],
            fences: [vk::Fence::null(); 2],
            slots: [Slot::Free, Slot::Free],
            staging: None,
        };
        match transfer.init(device) {
            Ok(()) => Ok(transfer),
            Err(error) => {
                transfer.destroy(device.raw());
                Err(error)
            }
        }
    }

    fn init(&mut self, device: &Device) -> Result<()> {
        let raw = device.raw();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device.physical().queue_family);
        // SAFETY: the device is alive and the family was opened with it.
        self.pool =
            unsafe { raw.create_command_pool(&pool_info, None) }.check("vkCreateCommandPool")?;
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);
        // SAFETY: the pool was created just above.
        let buffers =
            unsafe { raw.allocate_command_buffers(&allocate) }.check("vkAllocateCommandBuffers")?;
        self.command_buffers = [buffers[0], buffers[1]];
        for fence in &mut self.fences {
            // SAFETY: the device is alive.
            *fence = unsafe { raw.create_fence(&vk::FenceCreateInfo::default(), None) }
                .check("vkCreateFence")?;
        }
        self.staging = Some(HostMemory::create(raw, device.physical(), 2 * CHUNK)?);
        Ok(())
    }

    /// Destroys what exists of the context. Destroying a null handle is a
    /// no-op, so this also cleans up a context that failed half-way.
    pub(crate) fn destroy(mut self, raw: &ash::Device) {
        if let Some(staging) = self.staging.take() {
            staging.destroy(raw);
        }
        // SAFETY: nothing uses the objects any more (the caller waited for the
        // device); each is destroyed once, and destroying the pool frees the
        // command buffers allocated from it.
        unsafe {
            for fence in self.fences {
                raw.destroy_fence(fence, None);
            }
            raw.destroy_command_pool(self.pool, None);
        }
    }

    /// Waits for the chunk in `slot`, if any, and for a read copies its data
    /// out of the staging memory.
    fn finish(&mut self, device: &Device, slot: usize, host: &mut Host<'_>) -> Result<()> {
        let Slot::Busy(out) = std::mem::replace(&mut self.slots[slot], Slot::Free) else {
            return Ok(());
        };
        let raw = device.raw();
        let fence = self.fences[slot];
        // SAFETY: the fence belongs to this device and was submitted with the
        // chunk.
        unsafe { raw.wait_for_fences(&[fence], true, u64::MAX) }.check("vkWaitForFences")?;
        // SAFETY: the fence has signalled, so nothing waits on it.
        unsafe { raw.reset_fences(&[fence]) }.check("vkResetFences")?;
        if let (Some((start, len)), Host::Sink(sink)) = (out, host) {
            let staging = self.staging.as_ref().expect("staging exists once the context does");
            // SAFETY: the slot's `len` bytes lie inside the mapping and the
            // device finished writing them (the fence signalled, and the chunk
            // ends in a barrier that makes its writes visible to the host).
            let bytes =
                unsafe { std::slice::from_raw_parts(staging.ptr.as_ptr().add(slot * CHUNK), len) };
            sink[start..start + len].copy_from_slice(bytes);
        }
        Ok(())
    }

    /// Records and submits the copy of one chunk between its slot and `buffer`.
    fn submit_chunk(&mut self, device: &Device, buffer: vk::Buffer, chunk: Chunk) -> Result<()> {
        let Chunk { slot, buffer_offset, len, is_write, host_start } = chunk;
        let raw = device.raw();
        let cb = self.command_buffers[slot];
        let staging = self.staging.as_ref().expect("staging exists once the context does").buffer;
        let region = vk::BufferCopy::default()
            .src_offset(if is_write { (slot * CHUNK) as u64 } else { buffer_offset })
            .dst_offset(if is_write { buffer_offset } else { (slot * CHUNK) as u64 })
            .size(len as u64);
        let (src, dst) = if is_write { (staging, buffer) } else { (buffer, staging) };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer's previous submission has finished (its
        // slot was finished above), and the pool is used by this thread only
        // (the context is behind a mutex). Both handles are alive.
        unsafe {
            raw.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .check("vkResetCommandBuffer")?;
            raw.begin_command_buffer(cb, &begin).check("vkBeginCommandBuffer")?;
            // After everything submitted before: the data a read fetches may
            // have been written by a kernel, and a write must not overtake a
            // kernel that still reads the old contents.
            let before = full_barrier()
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            raw.cmd_pipeline_barrier(
                cb,
                WORK_STAGES,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            raw.cmd_copy_buffer(cb, src, dst, &[region]);
            let (dst_stage, dst_access) = if is_write {
                // Later kernels and copies on any stream see the data.
                (WORK_STAGES, full_barrier().dst_access_mask)
            } else {
                (vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ)
            };
            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(dst_access);
            raw.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
            raw.end_command_buffer(cb).check("vkEndCommandBuffer")?;
        }
        device.submit(&[cb], &[], &[], self.fences[slot])?;
        self.slots[slot] = Slot::Busy((!is_write).then_some((host_start, len)));
        Ok(())
    }

    fn run(
        &mut self,
        device: &Device,
        buffer: vk::Buffer,
        offset: u64,
        mut host: Host<'_>,
    ) -> Result<()> {
        let total = host.len();
        let is_write = matches!(host, Host::Source(_));
        let mut result = Ok(());
        let (mut done, mut index) = (0usize, 0usize);
        while done < total {
            let len = (total - done).min(CHUNK);
            let slot = index % 2;
            let step = self.finish(device, slot, &mut host).and_then(|()| {
                if let Host::Source(data) = &host {
                    let staging = self.staging.as_ref().expect("staging exists");
                    // SAFETY: the slot is free, so the device is not reading
                    // it, and `len` bytes at the slot's start lie inside the
                    // mapping.
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(staging.ptr.as_ptr().add(slot * CHUNK), len)
                    };
                    dst.copy_from_slice(&data[done..done + len]);
                }
                let chunk = Chunk {
                    slot,
                    buffer_offset: offset + done as u64,
                    len,
                    is_write,
                    host_start: done,
                };
                self.submit_chunk(device, buffer, chunk)
            });
            if let Err(error) = step {
                result = Err(error);
                break;
            }
            done += len;
            index += 1;
        }
        // Whatever happened, both slots must be idle before the call returns:
        // their command buffers and fences are reused by the next one.
        for slot in 0..2 {
            let drained = self.finish(device, slot, &mut host);
            result = result.and(drained);
        }
        result
    }
}

/// Copies `data` into `buffer` at `offset` and returns when it is there.
pub(crate) fn write(device: &Device, buffer: vk::Buffer, offset: u64, data: &[u8]) -> Result<()> {
    run(device, buffer, offset, Host::Source(data))
}

/// Copies from `buffer` at `offset` into `out` and returns when it is there.
pub(crate) fn read(device: &Device, buffer: vk::Buffer, offset: u64, out: &mut [u8]) -> Result<()> {
    run(device, buffer, offset, Host::Sink(out))
}

fn run(device: &Device, buffer: vk::Buffer, offset: u64, host: Host<'_>) -> Result<()> {
    let mut guard = lock(&device.0.transfer);
    if guard.is_none() {
        *guard = Some(Transfer::create(device)?);
    }
    let transfer = guard.as_mut().expect("created just above");
    transfer.run(device, buffer, offset, host)
}
