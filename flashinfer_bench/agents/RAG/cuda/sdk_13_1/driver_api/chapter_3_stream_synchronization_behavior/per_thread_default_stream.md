# Per-thread default stream

The per-thread default stream is an implicit stream local to both the thread and the CUcontext, and
which does not synchronize with other streams (just like explicitly created streams). The per-thread
default stream is not a non-blocking stream and will synchronize with the legacy default stream if both
are used in a program.

The per-thread default stream can be used explicitly with the CUstream (cudaStream_t) handle
CU_STREAM_PER_THREAD (cudaStreamPerThread).


CUDA Driver API TRM-06703-001 _vRelease Version  |  6