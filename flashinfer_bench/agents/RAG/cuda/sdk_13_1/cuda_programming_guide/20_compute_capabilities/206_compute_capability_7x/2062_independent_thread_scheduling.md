# 20.6.2. Independent Thread Scheduling

### 20.6.2. Independent Thread Scheduling[](#independent-thread-scheduling "Permalink to this headline")

The **NVIDIA Volta GPU Architecture** introduces *Independent Thread Scheduling* among threads in a warp, enabling intra-warp synchronization patterns previously unavailable and simplifying code changes when porting CPU code. However, this can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity of previous hardware architectures.

Below are code patterns of concern and suggested corrective actions for Volta-safe code.

1. For applications using warp intrinsics (`__shfl*`, `__any`, `__all`, `__ballot`), it is necessary that developers port their code to the new, safe, synchronizing counterpart, with the `*_sync` suffix. The new warp intrinsics take in a mask of threads that explicitly define which lanes (threads of a warp) must participate in the warp intrinsic. See [Warp Vote Functions](#warp-vote-functions) and [Warp Shuffle Functions](#warp-shuffle-functions) for details.

Since the intrinsics are available with CUDA 9.0+, (if necessary) code can be executed conditionally with the following preprocessor macro:

```
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
// *_sync intrinsic
#endif
```

These intrinsics are available on all architectures, not just **NVIDIA Volta GPU Architecture** or **NVIDIA Turing GPU Architecture**, and in most cases a single code-base will suffice for all architectures. Note, however, that for *Pascal* and earlier architectures, all threads in mask must execute the same warp intrinsic instruction in convergence, and the union of all values in mask must be equal to the warp’s active mask. The following code pattern is valid on **NVIDIA Volta GPU Architecture**, but not on *Pascal* or earlier architectures.

> ```
> if (tid % warpSize < 16) {
>     ...
>     float swapped = __shfl_xor_sync(0xffffffff, val, 16);
>     ...
> } else {
>     ...
>     float swapped = __shfl_xor_sync(0xffffffff, val, 16);
>     ...
> }
> ```

The replacement for `__ballot(1)` is `__activemask()`. Note that threads within a warp can diverge even within a single code path. As a result, `__activemask()` and `__ballot(1)` may return only a subset of the threads on the current code path. The following invalid code example sets bit `i` of `output` to 1 when `data[i]` is greater than `threshold`. `__activemask()` is used in an attempt to enable cases where `dataLen` is not a multiple of 32.

> ```
> // Sets bit in output[] to 1 if the correspond element in data[i]
> // is greater than 'threshold', using 32 threads in a warp.
>
> for (int i = warpLane; i < dataLen; i += warpSize) {
>     unsigned active = __activemask();
>     unsigned bitPack = __ballot_sync(active, data[i] > threshold);
>     if (warpLane == 0) {
>         output[i / 32] = bitPack;
>     }
> }
> ```

This code is invalid because CUDA does not guarantee that the warp will diverge ONLY at the loop condition. When divergence happens for other reasons, conflicting results will be computed for the same 32-bit output element by different subsets of threads in the warp. A correct code might use a non-divergent loop condition together with `__ballot_sync()` to safely enumerate the set of threads in the warp participating in the threshold calculation as follows.

> ```
> for (int i = warpLane; i - warpLane < dataLen; i += warpSize) {
>     unsigned active = __ballot_sync(0xFFFFFFFF, i < dataLen);
>     if (i < dataLen) {
>         unsigned bitPack = __ballot_sync(active, data[i] > threshold);
>         if (warpLane == 0) {
>             output[i / 32] = bitPack;
>         }
>     }
> }
> ```

[Discovery Pattern](#discovery-pattern-cg) demonstrates a valid use case for `__activemask()`.

1. If applications have warp-synchronous codes, they will need to insert the new `__syncwarp()` warp-wide barrier synchronization instruction between any steps where data is exchanged between threads via global or shared memory. Assumptions that code is executed in lockstep or that reads/writes from separate threads are visible across a warp without synchronization are invalid.

   ```
   __shared__ float s_buff[BLOCK_SIZE];
   s_buff[tid] = val;
   __syncthreads();

   // Inter-warp reduction
   for (int i = BLOCK_SIZE / 2; i >= 32; i /= 2) {
       if (tid < i) {
           s_buff[tid] += s_buff[tid+i];
       }
       __syncthreads();
   }

   // Intra-warp reduction
   // Butterfly reduction simplifies syncwarp mask
   if (tid < 32) {
       float temp;
       temp = s_buff[tid ^ 16]; __syncwarp();
       s_buff[tid] += temp;     __syncwarp();
       temp = s_buff[tid ^ 8];  __syncwarp();
       s_buff[tid] += temp;     __syncwarp();
       temp = s_buff[tid ^ 4];  __syncwarp();
       s_buff[tid] += temp;     __syncwarp();
       temp = s_buff[tid ^ 2];  __syncwarp();
       s_buff[tid] += temp;     __syncwarp();
   }

   if (tid == 0) {
       *output = s_buff[0] + s_buff[1];
   }
   __syncthreads();
   ```
2. Although `__syncthreads()` has been consistently documented as synchronizing all threads in the thread block, *Pascal* and prior architectures could only enforce synchronization at the warp level. In certain cases, this allowed a barrier to succeed without being executed by every thread as long as at least some thread in every warp reached the barrier. Starting with **NVIDIA Volta GPU Architecture**, the CUDA built-in `__syncthreads()` and PTX instruction `bar.sync` (and their derivatives) are enforced per thread and thus will not succeed until reached by all non-exited threads in the block. Code exploiting the previous behavior will likely deadlock and must be modified to ensure that all non-exited threads reach the barrier.

The `racecheck` and `synccheck` tools provided by `compute-saniter` can help with locating violations.

To aid migration while implementing the above-mentioned corrective actions, developers can opt-in to the Pascal scheduling model that does not support independent thread scheduling. See [Application Compatibility](#application-compatibility) for details.