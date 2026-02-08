# Default stream

The default stream, used when 0 is passed as a cudaStream_t or by APIs that operate on a stream
implicitly, can be configured to have either legacy or per-thread synchronization behavior as described
below.

The behavior can be controlled per compilation unit with the --default-stream
nvcc option. Alternatively, per-thread behavior can be enabled by defining the
CUDA_API_PER_THREAD_DEFAULT_STREAM macro before including any CUDA headers. Either way,
the CUDA_API_PER_THREAD_DEFAULT_STREAM macro will be defined in compilation units using perthread synchronization behavior.