# enum CUdevice_attribute

Device properties

###### Values

**CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1**
Maximum number of threads per block
**CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2**
Maximum block dimension X
**CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3**
Maximum block dimension Y
**CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4**
Maximum block dimension Z
**CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5**
Maximum grid dimension X
**CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6**
Maximum grid dimension Y
**CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7**
Maximum grid dimension Z
**CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8**


CUDA Driver API TRM-06703-001 _vRelease Version  |  23


Modules


Maximum shared memory available per block in bytes
**CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8**
Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
**CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9**
Memory available on device for __constant__ variables in a CUDA C kernel in bytes
**CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10**
Warp size in threads
**CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11**
Maximum pitch in bytes allowed by memory copies
**CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12**
Maximum number of 32-bit registers available per block
**CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12**
Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
**CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13**
Typical clock frequency in kilohertz
**CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14**
Alignment requirement for textures
**CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15**
Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.
**CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16**
Number of multiprocessors on device
**CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17**
Specifies whether there is a run time limit on kernels
**CU_DEVICE_ATTRIBUTE_INTEGRATED = 18**
Device is integrated with host memory
**CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19**
Device can map host memory into CUDA address space
**CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20**
Compute mode (See CUcomputemode for details)
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21**
Maximum 1D texture width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22**
Maximum 2D texture width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23**
Maximum 2D texture height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24**
Maximum 3D texture width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25**
Maximum 3D texture height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26**
Maximum 3D texture depth
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27**


CUDA Driver API TRM-06703-001 _vRelease Version  |  24


Modules


Maximum 2D layered texture width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28**
Maximum 2D layered texture height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29**
Maximum layers in a 2D layered texture
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27**
Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28**
Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29**
Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
**CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30**
Alignment requirement for surfaces
**CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31**
Device can possibly execute multiple kernels concurrently
**CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32**
Device has ECC support enabled
**CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33**
PCI bus ID of the device
**CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34**
PCI device ID of the device
**CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35**
Device is using TCC driver model
**CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36**
Peak memory clock frequency in kilohertz
**CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37**
Global memory bus width in bits
**CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38**
Size of L2 cache in bytes
**CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39**
Maximum resident threads per multiprocessor
**CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40**
Number of asynchronous engines
**CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41**
Device shares a unified address space with the host
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42**
Maximum 1D layered texture width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43**
Maximum layers in a 1D layered texture
**CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44**
Deprecated, do not use.
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45**
Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set


CUDA Driver API TRM-06703-001 _vRelease Version  |  25


Modules


**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46**
Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47**
Alternate maximum 3D texture width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48**
Alternate maximum 3D texture height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49**
Alternate maximum 3D texture depth
**CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50**
PCI domain ID of the device
**CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51**
Pitch alignment requirement for textures
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52**
Maximum cubemap texture width/height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53**
Maximum cubemap layered texture width/height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54**
Maximum layers in a cubemap layered texture
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55**
Maximum 1D surface width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56**
Maximum 2D surface width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57**
Maximum 2D surface height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58**
Maximum 3D surface width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59**
Maximum 3D surface height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60**
Maximum 3D surface depth
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61**
Maximum 1D layered surface width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62**
Maximum layers in a 1D layered surface
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63**
Maximum 2D layered surface width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64**
Maximum 2D layered surface height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65**
Maximum layers in a 2D layered surface
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66**
Maximum cubemap surface width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67**


CUDA Driver API TRM-06703-001 _vRelease Version  |  26


Modules


Maximum cubemap layered surface width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68**
Maximum layers in a cubemap layered surface
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69**
Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or
cuDeviceGetTexture1DLinearMaxWidth() instead.
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70**
Maximum 2D linear texture width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71**
Maximum 2D linear texture height
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72**
Maximum 2D linear texture pitch in bytes
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73**
Maximum mipmapped 2D texture width
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74**
Maximum mipmapped 2D texture height
**CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75**
Major compute capability version number
**CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76**
Minor compute capability version number
**CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77**
Maximum mipmapped 1D texture width
**CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78**
Device supports stream priorities
**CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79**
Device supports caching globals in L1
**CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80**
Device supports caching locals in L1
**CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81**
Maximum shared memory available per multiprocessor in bytes
**CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82**
Maximum number of 32-bit registers available per multiprocessor
**CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83**
Device can allocate managed memory on this system
**CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84**
Device is on a multi-GPU board
**CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85**
Unique id for a group of devices on the same multi-GPU board
**CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86**
Link between the device and the host supports all native atomic operations
**CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87**
Ratio of single precision performance (in floating-point operations per second) to double precision
performance


CUDA Driver API TRM-06703-001 _vRelease Version  |  27


Modules


**CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88**
Device supports coherently accessing pageable memory without calling cudaHostRegister on it
**CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89**
Device can coherently access managed memory concurrently with the CPU
**CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90**
Device supports compute preemption.
**CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91**
Device can access host registered memory at the same virtual address as the CPU
**CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92**
Deprecated, along with v1 MemOps API, cuStreamBatchMemOp and related APIs are supported.
**CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93**
Deprecated, along with v1 MemOps API, 64-bit operations are supported in cuStreamBatchMemOp
and related APIs.
**CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94**
Deprecated, along with v1 MemOps API, CU_STREAM_WAIT_VALUE_NOR is supported.
**CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95**
Device supports launching cooperative kernels via cuLaunchCooperativeKernel
**CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96**
Deprecated, cuLaunchCooperativeKernelMultiDevice is deprecated.
**CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97**
Maximum optin shared memory per block
**CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98**
The CU_STREAM_WAIT_VALUE_FLUSH flag and the
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See
Stream Memory Operations for additional details.
**CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99**
Device supports host memory registration via cudaHostRegister.
**CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES**
**= 100**
Device accesses pageable memory via the host's page tables.
**CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101**
The host can directly access managed memory on the device without migration.
**CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102**
Deprecated, Use
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
**CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102**
Device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate,
cuMemMap and related APIs
**CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED =**
**103**
Device supports exporting memory to a posix file descriptor with
cuMemExportToShareableHandle, if requested via cuMemCreate
**CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104**


CUDA Driver API TRM-06703-001 _vRelease Version  |  28


Modules


Device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle,
if requested via cuMemCreate
**CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105**
Device supports exporting memory to a Win32 KMT handle with
cuMemExportToShareableHandle, if requested via cuMemCreate
**CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106**
Maximum number of blocks per multiprocessor
**CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107**
Device supports compression of memory
**CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108**
Maximum L2 persisting lines capacity setting in bytes.
**CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109**
Maximum value of CUaccessPolicyWindow::num_bytes.
**CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED =**
**110**
Device supports specifying the GPUDirect RDMA flag with cuMemCreate
**CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111**
Shared memory reserved by CUDA driver per block in bytes
**CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112**
Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
**CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113**
Device supports using the cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to
register memory that must be mapped as read-only to the GPU
**CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114**
External timeline semaphore interop is supported on the device
**CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115**
Device supports using the cuMemAllocAsync and cuMemPool family of APIs
**CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116**
Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see [https://docs.nvidia.com/](https://docs.nvidia.com/cuda/gpudirect-rdma)
[cuda/gpudirect-rdma for more information)](https://docs.nvidia.com/cuda/gpudirect-rdma)
**CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117**
The returned attribute shall be interpreted as a bitmask, where the individual bits are described by
the CUflushGPUDirectRDMAWritesOptions enum
**CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118**
GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope
indicated by the returned attribute. See CUGPUDirectRDMAWritesOrdering for the numerical
values returned here.
**CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119**
Handle types supported with mempool based IPC
**CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120**
Indicates device supports cluster launch
**CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121**
Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays


CUDA Driver API TRM-06703-001 _vRelease Version  |  29


Modules


**CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122**
64-bit operations are supported in cuStreamBatchMemOp and related MemOp APIs.
**CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123**

CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
**CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124**
Device supports buffer sharing with dma_buf mechanism.
**CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125**
Device supports IPC Events.
**CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126**
Number of memory domains the device supports.
**CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127**
Device supports accessing memory using Tensor Map.
**CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128**
Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or
requested with cuMemCreate()
**CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129**
Device supports unified function pointers.
**CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130**
NUMA configuration of a device: value is of type CUdeviceNumaConfig enum
**CU_DEVICE_ATTRIBUTE_NUMA_ID = 131**
NUMA node ID of the GPU memory
**CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132**
Device supports switch multicast and reduction operations.
**CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133**
Indicates if contexts created on this device will be shared via MPS
**CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134**
NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA.
**CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135**
Device supports CIG with D3D12.
**CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK = 136**
The returned valued shall be interpreted as a bitmask, where the individual bits are described by the
CUmemDecompressAlgorithm enum.
**CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH = 137**
The returned valued is the maximum length in bytes of a single decompress operation that is
allowed.
**CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED = 138**
Device supports CIG with Vulkan.
**CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID = 139**
The combined 16-bit PCI device ID and 16-bit PCI vendor ID.
**CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID = 140**
The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.
**CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED**
**= 141**


CUDA Driver API TRM-06703-001 _vRelease Version  |  30


Modules


Device supports HOST_NUMA location with the virtual memory management APIs like
cuMemCreate, cuMemMap and related APIs
**CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED = 142**
Device supports HOST_NUMA location with the cuMemAllocAsync and cuMemPool family of
APIs
**CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143**
Device supports HOST_NUMA location IPC between nodes in a multi-node system.
**CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED = 144**
Device suports HOST location with the cuMemAllocAsync and cuMemPool family of APIs
**CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED =**
**145**
Device supports HOST location with the virtual memory management APIs like cuMemCreate,
cuMemMap and related APIs
**CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED = 146**
Device supports page-locked host memory buffer sharing with dma_buf mechanism.
**CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED =**
**147**
Link between the device and the host supports only some native atomic operations
**CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED = 148**
Device supports atomic reduction operations in stream batch memory operations
**CU_DEVICE_ATTRIBUTE_MAX**