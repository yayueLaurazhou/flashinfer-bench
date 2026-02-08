# 10.29.1. Using TMA to transfer one-dimensional arrays

### 10.29.1. Using TMA to transfer one-dimensional arrays[](#using-tma-to-transfer-one-dimensional-arrays "Permalink to this headline")

This section demonstrates how to write a simple kernel that read-modify-writes a
one-dimensional array using TMA. This shows how to how to load and store data
using bulk-asynchronous copies, as well as how to synchronize threads of
execution with those copies.

The code of the kernel is included below. Some functionality requires inline PTX
assembly that is currently made available through [libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx.html).
The availability of these wrappers can be
checked with the following code:

```
#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__
```

The kernel goes through the following stages:

1. Initialize shared memory barrier.
2. Initiate bulk-asynchronous copy of a block of memory from global to shared memory.
3. Arrive and wait on the shared memory barrier.
4. Increment the shared memory buffer values.
5. Wait for shared memory writes to be visible to the subsequent bulk-asynchronous copy, i.e., order the shared memory writes in the [async proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#async-proxy) before the next step.
6. Initiate bulk-asynchronous copy of the buffer in shared memory to global memory.
7. Wait at end of kernel for bulk-asynchronous copy to have finished reading shared memory.

```
#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                      // a)
    ptx::fence_proxy_async(ptx::space_shared);   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    cuda::memcpy_async(
        smem_data, 
        data + offset, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);   // b)
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
```

**Barrier initialization**. The barrier is initialized with the number of
threads participating in the block. As a result, the barrier will flip only if
all threads have arrived on this barrier. Shared memory barriers are described
in more detail in [Asynchronous Data Copies using cuda::barrier](#memcpy-async-barrier).
To make the initialized barrier visible to subsequent bulk-asynchronous copies, the
`fence.proxy.async.shared::cta` instruction is used. This instruction ensures that
subsequent bulk-asynchronous copy operations operate on the initialized barrier.

**TMA read**. The bulk-asynchronous copy instruction directs the
hardware to copy a large chunk of data into shared memory, and to update the
[transaction count](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations)
of the shared memory barrier after completing the read. In general, issuing as
few bulk copies with as big a size as possible results in the best performance.
Because the copy can be performed asynchronously by the hardware, it is not
necessary to split the copy into smaller chunks.

The thread that initiates the bulk-asynchronous copy operation arrives at the barrier
using `mbarrier.expect_tx`. This is automatically performed by `cuda::memcpy_async`. This tells the barrier that the thread has
arrived and also how many bytes (tx / transactions) are expected to arrive. Only
a single thread has to update the expected transaction count. If multiple
threads update the transaction count, the expected transaction will be the sum
of the updates. The barrier will only flip once all threads have arrived **and**
all bytes have arrived. Once the barrier has flipped, the bytes are safe to read
from shared memory, both by the threads as well as by subsequent
bulk-asynchronous copies. More information about barrier transaction accounting
can be found in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations).

**Barrier wait**. Waiting for the barrier to flip is done using
`mbarrier.try_wait`. It can either return true, indicating that the wait is
over, or return false, which may mean that the wait timed out. The while loop
waits for completion, and retries on time-out.

**SMEM write and sync**. The increment of the buffer values reads and writes to shared
memory. To make the writes visible to subsequent bulk-asynchronous copies, the
`fence.proxy.async.shared::cta` instruction is used. This orders the writes to
shared memory before subsequent reads from bulk-asynchronous copy operations,
which read through the async proxy. So each thread first orders the writes to
objects in shared memory in the async proxy via the
`fence.proxy.async.shared::cta`, and these operations by all threads are
ordered before the async operation performed in thread 0 using
`__syncthreads()`.

**TMA write and sync**. The write from shared to global memory is again
initiated by a single thread. The completion of the write is not tracked by a
shared memory barrier. Instead, a thread-local mechanism is used. Multiple
writes can be batched into a so-called *bulk async-group*. Afterwards, the
thread can wait for all operations in this group to have completed reading from
shared memory (as in the code above) or to have completed writing to global
memory, making the writes visible to the initiating thread. For more information,
refer to the PTX ISA documentation of [cp.async.bulk.wait\_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).
Note that the bulk-asynchronous and non-bulk asynchronous copy instructions have
different async-groups: there exist both `cp.async.wait_group` and
`cp.async.bulk.wait_group` instructions.

The bulk-asynchronous instructions have specific alignment requirements on their source and
destination addresses. More information can be found in the table below.

Table 9 Alignment requirements for one-dimensional bulk-asynchronous operations in Compute Capability 9.0.[](#table-alignment-one-dim-tma "Permalink to this table")




| Address / Size | Alignment |
| --- | --- |
| Global memory address | Must be 16 byte aligned. |
| Shared memory address | Must be 16 byte aligned. |
| Shared memory barrier address | Must be 8 byte aligned (this is guaranteed by `cuda::barrier`). |
| Size of transfer | Must be a multiple of 16 bytes. |