# __host__cudaError_t cudaStreamAddCallback (cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags)

Add a callback to a compute stream.

##### Parameters

**stream**

  - Stream to add callback to
**callback**

  - The function to call once preceding stream operations are complete


CUDA Runtime API vRelease Version  |  49


Modules


**userData**

  - User specified data to be passed to the callback function
**flags**

  - Reserved for future use, must be 0

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorInvalidValue, cudaErrorNotSupported

##### Description

Adds a callback to be called on the host after all currently enqueued items in the stream have
completed. For each cudaStreamAddCallback call, a callback will be executed exactly once. The
callback will block later work in the stream until it is finished.

The callback may be passed cudaSuccess or an error code. In the event of a device error, all
subsequently executed callbacks will receive an appropriate cudaError_t.

Callbacks must not make any CUDA API calls. Attempting to use CUDA APIs may result in
cudaErrorNotPermitted. Callbacks must not perform any synchronization that may depend on
outstanding device work or other callbacks that are not mandated to run earlier. Callbacks without a
mandated order (in independent streams) execute in undefined order and may be serialized.

For the purposes of Unified Memory, callback execution makes a number of guarantees:

The callback stream is considered idle for the duration of the callback. Thus, for example, a

##### **‣**

callback may always use memory attached to the callback stream.
The start of execution of a callback has the same effect as synchronizing an event recorded in

##### **‣**

the same stream immediately prior to the callback. It thus synchronizes streams which have been
"joined" prior to the callback.
Adding device work to any stream does not have the effect of making the stream active until

##### **‣**

all preceding callbacks have executed. Thus, for example, a callback might use global attached
memory even if work has been added to another stream, if it has been properly ordered with an
event.
Completion of a callback does not cause a stream to become active except as described above. The

##### **‣**

callback stream will remain idle if no device work follows the callback, and will remain idle across
consecutive callbacks without device work in between. Thus, for example, stream synchronization
can be done by signaling from a callback at the end of the stream.


CUDA Runtime API vRelease Version  |  50


Modules











See also:

cudaStreamCreate, cudaStreamCreateWithFlags, cudaStreamQuery, cudaStreamSynchronize,
cudaStreamWaitEvent, cudaStreamDestroy, cudaMallocManaged, cudaStreamAttachMemAsync,
cudaLaunchHostFunc, cuStreamAddCallback