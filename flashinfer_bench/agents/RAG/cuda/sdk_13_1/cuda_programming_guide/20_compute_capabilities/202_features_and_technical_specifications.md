# 20.2. Features and Technical Specifications

## 20.2. Features and Technical Specifications[](#features-and-technical-specifications "Permalink to this headline")

Table 26 Feature Support per Compute Capability[](#features-and-technical-specifications-feature-support-per-compute-capability "Permalink to this table")









| **Feature Support** | **Compute Capability** | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| (Unlisted features are supported for all compute capabilities) | 7.x | 8.x | 9.0 | 10.0 | 11.0 | 12.0 |
| Atomic functions operating on 128-bit integer values in global memory ([Atomic Functions](#atomic-functions)) | No | | Yes | | | |
| Atomic functions operating on 128-bit integer values in shared memory ([Atomic Functions](#atomic-functions)) | No | | Yes | | | |
| Atomic addition operating on float2 and float4 floating point vectors in global memory ([atomicAdd()](#atomicadd)) | No | | Yes | | | |
| Bfloat16-precision floating-point operations: addition, subtraction, multiplication, comparison, warp shuffle functions, conversion | No | | Yes | | | |
| Hardware-accelerated `memcpy_async` ([Asynchronous Data Copies using cuda::pipeline](#memcpy-async-pipeline)) | No | | Yes | | | |
| Hardware-accelerated Split Arrive/Wait Barrier ([Asynchronous Barrier](#aw-barrier)) | No | Yes | | | | |
| L2 Cache Residency Management ([Device Memory L2 Access Management](#l2-access-intro)) | No | Yes | | | | |
| DPX Instructions for Accelerated Dynamic Programming | No | | Yes | | | |
| Distributed Shared Memory | No | | Yes | | | |
| Thread Block Cluster | No | | Yes | | | |
| Tensor Memory Accelerator (TMA) unit | No | | Yes | | | |

Note that the KB and K units used in the following table correspond to 1024 bytes (i.e., a KiB) and 1024 respectively.

Table 27 Technical Specifications per Compute Capability[](#features-and-technical-specifications-technical-specifications-per-compute-capability "Permalink to this table")












|  | **Compute Capability** | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Technical Specifications | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.0 | 11.0 | 12.0 |
| Maximum number of resident grids per device (Concurrent Kernel Execution) | 128 | | | | | | | | |
| Maximum dimensionality of grid of thread blocks | 3 | | | | | | | | |
| Maximum x -dimension of a grid of thread blocks | 231-1 | | | | | | | | |
| Maximum y- or z-dimension of a grid of thread blocks | 65535 | | | | | | | | |
| Maximum dimensionality of thread block | 3 | | | | | | | | |
| Maximum x- or y-dimensionality of a block | 1024 | | | | | | | | |
| Maximum z-dimension of a block | 64 | | | | | | | | |
| Maximum number of threads per block | 1024 | | | | | | | | |
| Warp size | 32 | | | | | | | | |
| Maximum number of resident blocks per SM | 16 | 32 | 16 | | 24 | 32 | | 24 | |
| Maximum number of resident warps per SM | 32 | 64 | 48 | | | 64 | | 48 | |
| Maximum number of resident threads per SM | 1024 | 2048 | 1536 | | | 2048 | | 1536 | |
| Number of 32-bit registers per SM | 64 K | | | | | | | | |
| Maximum number of 32-bit registers per thread block | 64 K | | | | | | | | |
| Maximum number of 32-bit registers per thread | 255 | | | | | | | | |
| Maximum amount of shared memory per SM | 64 KB | 164 KB | 100 KB | 164 KB | 100 KB | 228 KB | | | 100 KB |
| Maximum amount of shared memory per thread block [27](#fn33) | 64 KB | 163 KB | 99 KB | 163 KB | 99 KB | 227 KB | | | 99 KB |
| Number of shared memory banks | 32 | | | | | | | | |
| Maximum amount of local memory per thread | 512 KB | | | | | | | | |
| Constant memory size | 64 KB | | | | | | | | |
| Cache working set per SM for constant memory | 8 KB | | | | | | | | |
| Cache working set per SM for texture memory | 32 or 64 KB | 28 KB ~ 192 KB | 28 KB ~ 128 KB | 28 KB ~ 192 KB | 28 KB ~ 128 KB | 28 KB ~ 256 KB | | | 28 KB ~ 128 KB |
| Maximum width for a 1D texture object using a CUDA array | 131072 | | | | | | | | |
| Maximum width for a 1D texture object using linear memory | 228 | | | | | | | | |
| Maximum width and number of layers for a 1D layered texture object | 32768 x 2048 | | | | | | | | |
| Maximum width and height for a 2D texture object using a CUDA array | 131072 x 65536 | | | | | | | | |
| Maximum width and height for a 2D texture object using linear memory | 131072 x 65000 | | | | | | | | |
| Maximum width and height for a 2D texture object using a CUDA array supporting texture gather | 32768 x 32768 | | | | | | | | |
| Maximum width, height, and number of layers for a 2D layered texture object | 32768 x 32768 x 2048 | | | | | | | | |
| Maximum width, height, and depth for a 3D texture object using to a CUDA array | 16384 x 16384 x 16384 | | | | | | | | |
| Maximum width (and height) for a cubemap texture object | 32768 | | | | | | | | |
| Maximum width (and height) and number of layers for a cubemap layered texture object | 32768 x 2046 | | | | | | | | |
| Maximum number of textures that can be bound to a kernel | 256 | | | | | | | | |
| Maximum width for a 1D surface object using a CUDA array | 32768 | | | | | | | | |
| Maximum width and number of layers for a 1D layered surface object | 32768 x 2048 | | | | | | | | |
| Maximum width and height for a 2D surface object using a CUDA array | 131072 x 65536 | | | | | | | | |
| Maximum width, height, and number of layers for a 2D layered surface object | 32768 x 32768 x 1048 | | | | | | | | |
| Maximum width, height, and depth for a 3D surface object using a CUDA array | 16384 x 16384 x 16384 | | | | | | | | |
| Maximum width (and height) for a cubemap surface object using a CUDA array | 32768 | | | | | | | | |
| Maximum width (and height) and number of layers for a cubemap layered surface object | 32768 x 2046 | | | | | | | | |
| Maximum number of surfaces that can use a kernel | 32 | | | | | | | | |