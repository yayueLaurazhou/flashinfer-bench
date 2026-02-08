# enum cudaDeviceAttr

CUDA device attributes

##### Values

**cudaDevAttrMaxThreadsPerBlock = 1**
Maximum number of threads per block
**cudaDevAttrMaxBlockDimX = 2**
Maximum block dimension X
**cudaDevAttrMaxBlockDimY = 3**
Maximum block dimension Y
**cudaDevAttrMaxBlockDimZ = 4**
Maximum block dimension Z


CUDA Runtime API vRelease Version  |  528


Modules


**cudaDevAttrMaxGridDimX = 5**
Maximum grid dimension X
**cudaDevAttrMaxGridDimY = 6**
Maximum grid dimension Y
**cudaDevAttrMaxGridDimZ = 7**
Maximum grid dimension Z
**cudaDevAttrMaxSharedMemoryPerBlock = 8**
Maximum shared memory available per block in bytes
**cudaDevAttrTotalConstantMemory = 9**
Memory available on device for __constant__ variables in a CUDA C kernel in bytes
**cudaDevAttrWarpSize = 10**
Warp size in threads
**cudaDevAttrMaxPitch = 11**
Maximum pitch in bytes allowed by memory copies
**cudaDevAttrMaxRegistersPerBlock = 12**
Maximum number of 32-bit registers available per block
**cudaDevAttrClockRate = 13**
Peak clock frequency in kilohertz
**cudaDevAttrTextureAlignment = 14**
Alignment requirement for textures
**cudaDevAttrGpuOverlap = 15**
Device can possibly copy memory and execute a kernel concurrently
**cudaDevAttrMultiProcessorCount = 16**
Number of multiprocessors on device
**cudaDevAttrKernelExecTimeout = 17**
Specifies whether there is a run time limit on kernels
**cudaDevAttrIntegrated = 18**
Device is integrated with host memory
**cudaDevAttrCanMapHostMemory = 19**
Device can map host memory into CUDA address space
**cudaDevAttrComputeMode = 20**
Compute mode (See cudaComputeMode for details)
**cudaDevAttrMaxTexture1DWidth = 21**
Maximum 1D texture width
**cudaDevAttrMaxTexture2DWidth = 22**
Maximum 2D texture width
**cudaDevAttrMaxTexture2DHeight = 23**
Maximum 2D texture height
**cudaDevAttrMaxTexture3DWidth = 24**
Maximum 3D texture width
**cudaDevAttrMaxTexture3DHeight = 25**
Maximum 3D texture height
**cudaDevAttrMaxTexture3DDepth = 26**


CUDA Runtime API vRelease Version  |  529


Modules


Maximum 3D texture depth
**cudaDevAttrMaxTexture2DLayeredWidth = 27**
Maximum 2D layered texture width
**cudaDevAttrMaxTexture2DLayeredHeight = 28**
Maximum 2D layered texture height
**cudaDevAttrMaxTexture2DLayeredLayers = 29**
Maximum layers in a 2D layered texture
**cudaDevAttrSurfaceAlignment = 30**
Alignment requirement for surfaces
**cudaDevAttrConcurrentKernels = 31**
Device can possibly execute multiple kernels concurrently
**cudaDevAttrEccEnabled = 32**
Device has ECC support enabled
**cudaDevAttrPciBusId = 33**
PCI bus ID of the device
**cudaDevAttrPciDeviceId = 34**
PCI device ID of the device
**cudaDevAttrTccDriver = 35**
Device is using TCC driver model
**cudaDevAttrMemoryClockRate = 36**
Peak memory clock frequency in kilohertz
**cudaDevAttrGlobalMemoryBusWidth = 37**
Global memory bus width in bits
**cudaDevAttrL2CacheSize = 38**
Size of L2 cache in bytes
**cudaDevAttrMaxThreadsPerMultiProcessor = 39**
Maximum resident threads per multiprocessor
**cudaDevAttrAsyncEngineCount = 40**
Number of asynchronous engines
**cudaDevAttrUnifiedAddressing = 41**
Device shares a unified address space with the host
**cudaDevAttrMaxTexture1DLayeredWidth = 42**
Maximum 1D layered texture width
**cudaDevAttrMaxTexture1DLayeredLayers = 43**
Maximum layers in a 1D layered texture
**cudaDevAttrMaxTexture2DGatherWidth = 45**
Maximum 2D texture width if cudaArrayTextureGather is set
**cudaDevAttrMaxTexture2DGatherHeight = 46**
Maximum 2D texture height if cudaArrayTextureGather is set
**cudaDevAttrMaxTexture3DWidthAlt = 47**
Alternate maximum 3D texture width
**cudaDevAttrMaxTexture3DHeightAlt = 48**
Alternate maximum 3D texture height


CUDA Runtime API vRelease Version  |  530


Modules


**cudaDevAttrMaxTexture3DDepthAlt = 49**
Alternate maximum 3D texture depth
**cudaDevAttrPciDomainId = 50**
PCI domain ID of the device
**cudaDevAttrTexturePitchAlignment = 51**
Pitch alignment requirement for textures
**cudaDevAttrMaxTextureCubemapWidth = 52**
Maximum cubemap texture width/height
**cudaDevAttrMaxTextureCubemapLayeredWidth = 53**
Maximum cubemap layered texture width/height
**cudaDevAttrMaxTextureCubemapLayeredLayers = 54**
Maximum layers in a cubemap layered texture
**cudaDevAttrMaxSurface1DWidth = 55**
Maximum 1D surface width
**cudaDevAttrMaxSurface2DWidth = 56**
Maximum 2D surface width
**cudaDevAttrMaxSurface2DHeight = 57**
Maximum 2D surface height
**cudaDevAttrMaxSurface3DWidth = 58**
Maximum 3D surface width
**cudaDevAttrMaxSurface3DHeight = 59**
Maximum 3D surface height
**cudaDevAttrMaxSurface3DDepth = 60**
Maximum 3D surface depth
**cudaDevAttrMaxSurface1DLayeredWidth = 61**
Maximum 1D layered surface width
**cudaDevAttrMaxSurface1DLayeredLayers = 62**
Maximum layers in a 1D layered surface
**cudaDevAttrMaxSurface2DLayeredWidth = 63**
Maximum 2D layered surface width
**cudaDevAttrMaxSurface2DLayeredHeight = 64**
Maximum 2D layered surface height
**cudaDevAttrMaxSurface2DLayeredLayers = 65**
Maximum layers in a 2D layered surface
**cudaDevAttrMaxSurfaceCubemapWidth = 66**
Maximum cubemap surface width
**cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67**
Maximum cubemap layered surface width
**cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68**
Maximum layers in a cubemap layered surface
**cudaDevAttrMaxTexture1DLinearWidth = 69**
Maximum 1D linear texture width
**cudaDevAttrMaxTexture2DLinearWidth = 70**


CUDA Runtime API vRelease Version  |  531


Modules


Maximum 2D linear texture width
**cudaDevAttrMaxTexture2DLinearHeight = 71**
Maximum 2D linear texture height
**cudaDevAttrMaxTexture2DLinearPitch = 72**
Maximum 2D linear texture pitch in bytes
**cudaDevAttrMaxTexture2DMipmappedWidth = 73**
Maximum mipmapped 2D texture width
**cudaDevAttrMaxTexture2DMipmappedHeight = 74**
Maximum mipmapped 2D texture height
**cudaDevAttrComputeCapabilityMajor = 75**
Major compute capability version number
**cudaDevAttrComputeCapabilityMinor = 76**
Minor compute capability version number
**cudaDevAttrMaxTexture1DMipmappedWidth = 77**
Maximum mipmapped 1D texture width
**cudaDevAttrStreamPrioritiesSupported = 78**
Device supports stream priorities
**cudaDevAttrGlobalL1CacheSupported = 79**
Device supports caching globals in L1
**cudaDevAttrLocalL1CacheSupported = 80**
Device supports caching locals in L1
**cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81**
Maximum shared memory available per multiprocessor in bytes
**cudaDevAttrMaxRegistersPerMultiprocessor = 82**
Maximum number of 32-bit registers available per multiprocessor
**cudaDevAttrManagedMemory = 83**
Device can allocate managed memory on this system
**cudaDevAttrIsMultiGpuBoard = 84**
Device is on a multi-GPU board
**cudaDevAttrMultiGpuBoardGroupID = 85**
Unique identifier for a group of devices on the same multi-GPU board
**cudaDevAttrHostNativeAtomicSupported = 86**
Link between the device and the host supports native atomic operations
**cudaDevAttrSingleToDoublePrecisionPerfRatio = 87**
Ratio of single precision performance (in floating-point operations per second) to double precision
performance
**cudaDevAttrPageableMemoryAccess = 88**
Device supports coherently accessing pageable memory without calling cudaHostRegister on it
**cudaDevAttrConcurrentManagedAccess = 89**
Device can coherently access managed memory concurrently with the CPU
**cudaDevAttrComputePreemptionSupported = 90**
Device supports Compute Preemption
**cudaDevAttrCanUseHostPointerForRegisteredMem = 91**


CUDA Runtime API vRelease Version  |  532


Modules


Device can access host registered memory at the same virtual address as the CPU
**cudaDevAttrReserved92 = 92**
**cudaDevAttrReserved93 = 93**
**cudaDevAttrReserved94 = 94**
**cudaDevAttrCooperativeLaunch = 95**
Device supports launching cooperative kernels via cudaLaunchCooperativeKernel
**cudaDevAttrReserved96 = 96**
**cudaDevAttrMaxSharedMemoryPerBlockOptin = 97**
The maximum optin shared memory per block. This value may vary by chip. See
cudaFuncSetAttribute
**cudaDevAttrCanFlushRemoteWrites = 98**
Device supports flushing of outstanding remote writes.
**cudaDevAttrHostRegisterSupported = 99**
Device supports host memory registration via cudaHostRegister.
**cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100**
Device accesses pageable memory via the host's page tables.
**cudaDevAttrDirectManagedMemAccessFromHost = 101**
Host can directly access managed memory on the device without migration.
**cudaDevAttrMaxBlocksPerMultiprocessor = 106**
Maximum number of blocks per multiprocessor
**cudaDevAttrMaxPersistingL2CacheSize = 108**
Maximum L2 persisting lines capacity setting in bytes.
**cudaDevAttrMaxAccessPolicyWindowSize = 109**
Maximum value of cudaAccessPolicyWindow::num_bytes.
**cudaDevAttrReservedSharedMemoryPerBlock = 111**
Shared memory reserved by CUDA driver per block in bytes
**cudaDevAttrSparseCudaArraySupported = 112**
Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
**cudaDevAttrHostRegisterReadOnlySupported = 113**
Device supports using the cudaHostRegister flag cudaHostRegisterReadOnly to register memory
that must be mapped as read-only to the GPU
**cudaDevAttrTimelineSemaphoreInteropSupported = 114**
External timeline semaphore interop is supported on the device
**cudaDevAttrMemoryPoolsSupported = 115**
Device supports using the cudaMallocAsync and cudaMemPool family of APIs
**cudaDevAttrGPUDirectRDMASupported = 116**
Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see [https://docs.nvidia.com/](https://docs.nvidia.com/cuda/gpudirect-rdma)
[cuda/gpudirect-rdma for more information)](https://docs.nvidia.com/cuda/gpudirect-rdma)
**cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117**
The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the
cudaFlushGPUDirectRDMAWritesOptions enum
**cudaDevAttrGPUDirectRDMAWritesOrdering = 118**


CUDA Runtime API vRelease Version  |  533


Modules


GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope
indicated by the returned attribute. See cudaGPUDirectRDMAWritesOrdering for the numerical
values returned here.
**cudaDevAttrMemoryPoolSupportedHandleTypes = 119**
Handle types supported with mempool based IPC
**cudaDevAttrClusterLaunch = 120**
Indicates device supports cluster launch
**cudaDevAttrDeferredMappingCudaArraySupported = 121**
Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
**cudaDevAttrReserved122 = 122**
**cudaDevAttrReserved123 = 123**
**cudaDevAttrReserved124 = 124**
**cudaDevAttrIpcEventSupport = 125**
Device supports IPC Events.
**cudaDevAttrMemSyncDomainCount = 126**
Number of memory synchronization domains the device supports.
**cudaDevAttrReserved127 = 127**
**cudaDevAttrReserved128 = 128**
**cudaDevAttrReserved129 = 129**
**cudaDevAttrNumaConfig = 130**
NUMA configuration of a device: value is of type cudaDeviceNumaConfig enum
**cudaDevAttrNumaId = 131**
NUMA node ID of the GPU memory
**cudaDevAttrReserved132 = 132**
**cudaDevAttrMpsEnabled = 133**
Contexts created on this device will be shared via MPS
**cudaDevAttrHostNumaId = 134**
NUMA ID of the host node closest to the device or -1 when system does not support NUMA
**cudaDevAttrD3D12CigSupported = 135**
Device supports CIG with D3D12.
**cudaDevAttrVulkanCigSupported = 138**
Device supports CIG with Vulkan.
**cudaDevAttrGpuPciDeviceId = 139**
The combined 16-bit PCI device ID and 16-bit PCI vendor ID.
**cudaDevAttrGpuPciSubsystemId = 140**
The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.
**cudaDevAttrReserved141 = 141**
**cudaDevAttrHostNumaMemoryPoolsSupported = 142**
Device supports HOST_NUMA location with the cudaMallocAsync and cudaMemPool family of
APIs
**cudaDevAttrHostNumaMultinodeIpcSupported = 143**
Device supports HostNuma location IPC between nodes in a multi-node system.
**cudaDevAttrHostMemoryPoolsSupported = 144**


CUDA Runtime API vRelease Version  |  534


Modules


Device suports HOST location with the cuMemAllocAsync and cuMemPool family of APIs
**cudaDevAttrReserved145 = 145**
**cudaDevAttrOnlyPartialHostNativeAtomicSupported = 147**
Link between the device and the host supports only some native atomic operations
**cudaDevAttrMax**