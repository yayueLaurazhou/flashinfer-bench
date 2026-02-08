# Legacy default stream

The legacy default stream is an implicit stream which synchronizes with all other streams in the same
CUcontext except for non-blocking streams, described below. (For applications using the runtime
APIs only, there will be one context per device.) When an action is taken in the legacy stream such as a
kernel launch or cudaStreamWaitEvent(), the legacy stream first waits on all blocking streams, the
action is queued in the legacy stream, and then all blocking streams wait on the legacy stream.

For example, the following code launches a kernel k_1 in stream s, then k_2 in the legacy stream, then
k_3 in stream s:


k_1<<<1, 1, 0, s>>>();
k_2<<<1, 1>>>();
k_3<<<1, 1, 0, s>>>();

The resulting behavior is that k_2 will block on k_1 and k_3 will block on k_2.

Non-blocking streams which do not synchronize with the legacy stream can be created using the
cudaStreamNonBlocking flag with the stream creation APIs.

The legacy default stream can be used explicitly with the CUstream (cudaStream_t) handle
CU_STREAM_LEGACY (cudaStreamLegacy).


CUDA Driver API TRM-06703-001 _vRelease Version  |  5


Stream synchronization behavior