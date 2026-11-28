//! The part of the CUDA driver API the engine uses, bound at run time.
//!
//! `libcuda.so.1` ships with the NVIDIA driver, not with the toolkit, so
//! binding it with `dlopen` lets one binary start on machines without CUDA
//! (and report that) instead of failing to link. Signatures follow `cuda.h`
//! of CUDA 13; where the header maps a name to a versioned symbol
//! (`cuMemAlloc` -> `cuMemAlloc_v2`), the versioned symbol is bound.

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::{c_char, c_int, c_uint, c_void};

use libloading::Library;

pub(crate) type CUresult = c_int;
pub(crate) type CUdevice = c_int;
pub(crate) type CUdeviceptr = u64;
pub(crate) type CUcontext = *mut c_void;
pub(crate) type CUmodule = *mut c_void;
pub(crate) type CUfunction = *mut c_void;
pub(crate) type CUstream = *mut c_void;
pub(crate) type CUevent = *mut c_void;

pub(crate) type CUmemGenericAllocationHandle = u64;

/// `CUmemLocation`: a device by its ordinal (`CU_MEM_LOCATION_TYPE_DEVICE`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUmemLocation {
    pub kind: c_int,
    pub id: c_int,
}

/// `CUmemAllocationProp`: memory of one device, no handle to share it.
#[repr(C)]
pub(crate) struct CUmemAllocationProp {
    pub kind: c_int,
    pub requested_handle_types: c_int,
    pub location: CUmemLocation,
    pub win32_handle_meta_data: *mut c_void,
    /// `allocFlags`: compression type, RDMA, usage, reserved; all 0.
    pub alloc_flags: [u8; 8],
}

/// `CUmemAccessDesc`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: c_int,
}

pub(crate) const CU_MEM_ALLOCATION_TYPE_PINNED: c_int = 1;
pub(crate) const CU_MEM_LOCATION_TYPE_DEVICE: c_int = 1;
pub(crate) const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: c_int = 3;
pub(crate) const CU_MEM_ALLOC_GRANULARITY_MINIMUM: c_int = 0;

pub(crate) const CUDA_SUCCESS: CUresult = 0;
pub(crate) const CUDA_ERROR_NOT_READY: CUresult = 600;
pub(crate) const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: CUresult = 704;
pub(crate) const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: CUresult = 708;

pub(crate) const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub(crate) const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: c_int = 33;
pub(crate) const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: c_int = 34;
pub(crate) const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
pub(crate) const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub(crate) const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: c_int = 50;
pub(crate) const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: c_int = 38;
pub(crate) const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

pub(crate) const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
pub(crate) const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
pub(crate) const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
pub(crate) const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

pub(crate) const CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: c_int = 1;
pub(crate) const CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED: c_int = 2;

pub(crate) const CU_MEMHOSTALLOC_PORTABLE: c_uint = 1;
pub(crate) const CU_MEMHOSTREGISTER_PORTABLE: c_uint = 1;
pub(crate) const CU_MEMHOSTREGISTER_DEVICEMAP: c_uint = 2;
pub(crate) const CU_MEMHOSTREGISTER_READ_ONLY: c_uint = 8;
pub(crate) const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub(crate) const CU_EVENT_DISABLE_TIMING: c_uint = 2;

macro_rules! driver_api {
    ($( fn $name:ident = $symbol:literal ( $($arg:ty),* ); )*) => {
        /// The bound driver functions. The library stays loaded for as long
        /// as this table exists, which keeps the function pointers valid.
        pub(crate) struct Api {
            _library: Library,
            $( pub(crate) $name: unsafe extern "C" fn($($arg),*) -> CUresult, )*
        }

        impl Api {
            /// Binds every function, or returns the name of the first symbol
            /// that is missing.
            ///
            /// # Safety
            ///
            /// `library` must be the CUDA driver library; the symbols are
            /// trusted to have the signatures declared here.
            pub(crate) unsafe fn bind(library: Library) -> Result<Self, &'static str> {
                $(
                    // SAFETY: the caller guarantees this is libcuda, whose
                    // symbol of this name has this signature (cuda.h).
                    let $name = unsafe {
                        *library
                            .get::<unsafe extern "C" fn($($arg),*) -> CUresult>(
                                concat!($symbol, "\0").as_bytes(),
                            )
                            .map_err(|_| $symbol)?
                    };
                )*
                Ok(Self { _library: library, $($name),* })
            }
        }
    };
}

driver_api! {
    fn init = "cuInit"(c_uint);
    fn driver_get_version = "cuDriverGetVersion"(*mut c_int);
    fn get_error_name = "cuGetErrorName"(CUresult, *mut *const c_char);
    fn get_error_string = "cuGetErrorString"(CUresult, *mut *const c_char);

    fn device_get_count = "cuDeviceGetCount"(*mut c_int);
    fn device_get = "cuDeviceGet"(*mut CUdevice, c_int);
    fn device_get_name = "cuDeviceGetName"(*mut c_char, c_int, CUdevice);
    fn device_get_uuid = "cuDeviceGetUuid_v2"(*mut [u8; 16], CUdevice);
    fn device_total_mem = "cuDeviceTotalMem_v2"(*mut usize, CUdevice);
    fn device_get_attribute = "cuDeviceGetAttribute"(*mut c_int, c_int, CUdevice);
    fn device_can_access_peer = "cuDeviceCanAccessPeer"(*mut c_int, CUdevice, CUdevice);
    fn device_get_p2p_attribute = "cuDeviceGetP2PAttribute"(*mut c_int, c_int, CUdevice, CUdevice);

    fn primary_ctx_set_flags = "cuDevicePrimaryCtxSetFlags_v2"(CUdevice, c_uint);
    fn primary_ctx_retain = "cuDevicePrimaryCtxRetain"(*mut CUcontext, CUdevice);
    fn primary_ctx_release = "cuDevicePrimaryCtxRelease_v2"(CUdevice);
    fn ctx_set_current = "cuCtxSetCurrent"(CUcontext);
    fn ctx_synchronize = "cuCtxSynchronize"();
    fn ctx_enable_peer_access = "cuCtxEnablePeerAccess"(CUcontext, c_uint);

    fn mem_get_info = "cuMemGetInfo_v2"(*mut usize, *mut usize);
    fn mem_alloc = "cuMemAlloc_v2"(*mut CUdeviceptr, usize);
    fn mem_free = "cuMemFree_v2"(CUdeviceptr);
    fn mem_host_alloc = "cuMemHostAlloc"(*mut *mut c_void, usize, c_uint);
    fn mem_free_host = "cuMemFreeHost"(*mut c_void);
    fn mem_host_register = "cuMemHostRegister_v2"(*mut c_void, usize, c_uint);
    fn mem_host_unregister = "cuMemHostUnregister"(*mut c_void);
    fn mem_host_get_device_pointer = "cuMemHostGetDevicePointer_v2"(
        *mut CUdeviceptr, *mut c_void, c_uint
    );

    fn mem_get_allocation_granularity = "cuMemGetAllocationGranularity"(
        *mut usize, *const CUmemAllocationProp, c_int
    );
    fn mem_address_reserve = "cuMemAddressReserve"(*mut CUdeviceptr, usize, usize, CUdeviceptr, u64);
    fn mem_address_free = "cuMemAddressFree"(CUdeviceptr, usize);
    fn mem_create = "cuMemCreate"(
        *mut CUmemGenericAllocationHandle, usize, *const CUmemAllocationProp, u64
    );
    fn mem_release = "cuMemRelease"(CUmemGenericAllocationHandle);
    fn mem_map = "cuMemMap"(CUdeviceptr, usize, usize, CUmemGenericAllocationHandle, u64);
    fn mem_unmap = "cuMemUnmap"(CUdeviceptr, usize);
    fn mem_set_access = "cuMemSetAccess"(CUdeviceptr, usize, *const CUmemAccessDesc, usize);

    fn memcpy_htod = "cuMemcpyHtoD_v2"(CUdeviceptr, *const c_void, usize);
    fn memcpy_dtoh = "cuMemcpyDtoH_v2"(*mut c_void, CUdeviceptr, usize);
    fn memcpy_htod_async = "cuMemcpyHtoDAsync_v2"(CUdeviceptr, *const c_void, usize, CUstream);
    fn memcpy_dtoh_async = "cuMemcpyDtoHAsync_v2"(*mut c_void, CUdeviceptr, usize, CUstream);
    fn memcpy_dtod_async = "cuMemcpyDtoDAsync_v2"(CUdeviceptr, CUdeviceptr, usize, CUstream);
    fn memcpy_async = "cuMemcpyAsync"(CUdeviceptr, CUdeviceptr, usize, CUstream);
    fn memcpy_peer_async = "cuMemcpyPeerAsync"(
        CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize, CUstream
    );
    fn memset_d8_async = "cuMemsetD8Async"(CUdeviceptr, u8, usize, CUstream);

    fn module_load_data = "cuModuleLoadData"(*mut CUmodule, *const c_void);
    fn module_unload = "cuModuleUnload"(CUmodule);
    fn module_get_function = "cuModuleGetFunction"(*mut CUfunction, CUmodule, *const c_char);
    fn launch_kernel = "cuLaunchKernel"(
        CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream,
        *mut *mut c_void, *mut *mut c_void
    );

    fn func_set_attribute = "cuFuncSetAttribute"(CUfunction, c_int, c_int);
    fn func_get_attribute = "cuFuncGetAttribute"(*mut c_int, c_int, CUfunction);
    fn occupancy_max_active_blocks =
        "cuOccupancyMaxActiveBlocksPerMultiprocessor"(*mut c_int, CUfunction, c_int, usize);

    fn stream_create = "cuStreamCreateWithPriority"(*mut CUstream, c_uint, c_int);
    fn stream_destroy = "cuStreamDestroy_v2"(CUstream);
    fn stream_synchronize = "cuStreamSynchronize"(CUstream);
    fn stream_query = "cuStreamQuery"(CUstream);
    fn stream_wait_event = "cuStreamWaitEvent"(CUstream, CUevent, c_uint);

    fn event_create = "cuEventCreate"(*mut CUevent, c_uint);
    fn event_destroy = "cuEventDestroy_v2"(CUevent);
    fn event_record = "cuEventRecord"(CUevent, CUstream);
    fn event_query = "cuEventQuery"(CUevent);
    fn event_synchronize = "cuEventSynchronize"(CUevent);
    fn event_elapsed_time = "cuEventElapsedTime_v2"(*mut f32, CUevent, CUevent);
}
