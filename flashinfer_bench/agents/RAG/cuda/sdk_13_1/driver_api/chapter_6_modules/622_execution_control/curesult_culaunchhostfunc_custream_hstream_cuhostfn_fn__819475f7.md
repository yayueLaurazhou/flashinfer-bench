# CUresult cuLaunchHostFunc (CUstream hStream, CUhostFn fn, void *userData)

Enqueues a host function call in a stream.

###### Parameters

**hStream**

  - Stream to enqueue function call in
**fn**

  - The function to call once preceding stream operations are complete
**userData**

  - User-specified data to be passed to the function

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Enqueues a host function to run in a stream. The function will be called after currently enqueued work
and will block work added after it.

The host function must not make any CUDA API calls. Attempting to use a CUDA API may result
in CUDA_ERROR_NOT_PERMITTED, but this is not required. The host function must not perform
any synchronization that may depend on outstanding CUDA work not mandated to run earlier. Host
functions without a mandated order (such as in independent streams) execute in undefined order and
may be serialized.

For the purposes of Unified Memory, execution makes a number of guarantees:

The stream is considered idle for the duration of the function's execution. Thus, for example, the

###### **‣**

function may always use memory attached to the stream it was enqueued in.


CUDA Driver API TRM-06703-001 _vRelease Version  |  395


Modules


The start of execution of the function has the same effect as synchronizing an event recorded in

###### **‣**

the same stream immediately prior to the function. It thus synchronizes streams which have been
"joined" prior to the function.
Adding device work to any stream does not have the effect of making the stream active until all

###### **‣**

preceding host functions and stream callbacks have executed. Thus, for example, a function might
use global attached memory even if work has been added to another stream, if the work has been
ordered behind the function call with an event.
Completion of the function does not cause a stream to become active except as described above.

###### **‣**

The stream will remain idle if no device work follows the function, and will remain idle across
consecutive host functions or stream callbacks without device work in between. Thus, for example,
stream synchronization can be done by signaling from a host function at the end of the stream.

Note that, in contrast to cuStreamAddCallback, the function will not be called in the event of an error
in the CUDA context.





See also:

cuStreamCreate, cuStreamQuery, cuStreamSynchronize, cuStreamWaitEvent, cuStreamDestroy,
cuMemAllocManaged, cuStreamAttachMemAsync, cuStreamAddCallback