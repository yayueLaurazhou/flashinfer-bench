# **‣**

context.

Note: Developers should treat cudaExecutionContext_t as an opaque handle and avoid assumptions
about its underlying representation. The CUDA Runtime does not provide a way to convert this handle
into a CUcontext or CUgreenCtx.

Primary Context (aka Device Execution Context)


CUDA Runtime API vRelease Version  |  515


Modules


The primary context is the default execution context associated with a device in the Runtime. It can be
obtained via a call to cudaDeviceGetExecutionCtx(). There is a one-to-one mapping between CUDA
devices in the runtime and their primary contexts within a process.

From the CUDA Runtime’s perspective, a device and its primary context are functionally synonymous.

Unless explicitly overridden, either by making a different context current via the Driver API (e.g.,
cuCtxSetCurrent()) or by using an explicit execution context handle, the Runtime will implicitly
initialize and use the primary context for API calls as needed.

Initialization and Tear-Down

Unless an explicit execution context is specified (see “Execution Context Management” for APIs),
CUDA Runtime API calls operate on the CUDA Driver CUcontext which is current to the calling
host thread. If no CUcontext is current to the calling thread when a CUDA Runtime API call
which requires an active context is made, then the primary context (device execution context) for
a device will be selected, made current to the calling thread, and initialized. The context will be
initialized using the parameters specified by the CUDA Runtime API functions cudaSetDeviceFlags(),
cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice(), cudaD3D11SetDirect3DDevice(),
cudaGLSetGLDevice(), and cudaVDPAUSetVDPAUDevice(). Note that these functions will fail with
cudaErrorSetOnActiveProcess if they are called when the primary context for the specified device has
already been initialized, except for cudaSetDeviceFlags() which will simply overwrite the previous
settings.

The function cudaInitDevice() ensures that the primary context is initialized for the requested device
but does not make it current to the calling thread.

The function cudaSetDevice() initializes the primary context for the specified device and makes it
current to the calling thread by calling cuCtxSetCurrent().

Primary contexts will remain active until they are explicitly deinitialized using cudaDeviceReset().
The function cudaDeviceReset() will deinitialize the primary context for the calling thread's current
device immediately. The context will remain current to all of the threads that it was current to. The next
CUDA Runtime API call on any thread which requires an active context will trigger the reinitialization
of that device's primary context.

Note that primary contexts are shared resources. It is recommended that the primary context not be
reset except just before exit or to recover from an unspecified launch failure.

CUcontext Interoperability

Note that the use of multiple CUcontext s per device within a single process will substantially degrade
performance and is strongly discouraged. Instead, it is highly recommended to either use execution
contexts cudaExecutionContext_t or the implicit one-to-one device-to-primary context mapping for the
process provided by the CUDA Runtime API.

If a non-primary CUcontext created by the CUDA Driver API is current to a thread then the CUDA
Runtime API calls to that thread will operate on that CUcontext, with some exceptions listed below.
Interoperability between data types is discussed in the following sections.


CUDA Runtime API vRelease Version  |  516


Modules


The function cudaDeviceEnablePeerAccess() and the rest of the peer access API may not be called
when a non-primary CUcontext is current. To use the peer access APIs with a context created using the
CUDA Driver API, it is necessary that the CUDA Driver API be used to access these features.

All CUDA Runtime API state (e.g, global variables' addresses and values) travels with its underlying
CUcontext. In particular, if a CUcontext is moved from one thread to another then all CUDA Runtime
API state will move to that thread as well.

Please note that attaching to legacy CUcontext (those with a version of 3010 as
returned by cuCtxGetApiVersion()) is not possible. The CUDA Runtime will return
cudaErrorIncompatibleDriverContext in such cases.

Interactions between CUstream and cudaStream_t

The types CUstream and cudaStream_t are identical and may be used interchangeably.

Interactions between CUevent and cudaEvent_t

The types CUevent and cudaEvent_t are identical and may be used interchangeably.

Interactions between CUarray and cudaArray_t

The types CUarray and struct cudaArray * represent the same data type and may be used
interchangeably by casting the two types between each other.

In order to use a CUarray in a CUDA Runtime API function which takes a struct cudaArray *, it is
necessary to explicitly cast the CUarray to a struct cudaArray *.

In order to use a struct cudaArray * in a CUDA Driver API function which takes a CUarray, it is
necessary to explicitly cast the struct cudaArray * to a CUarray .

Interactions between CUgraphicsResource and cudaGraphicsResource_t

The types CUgraphicsResource and cudaGraphicsResource_t represent the same data type and may be
used interchangeably by casting the two types between each other.

In order to use a CUgraphicsResource in a CUDA Runtime API function which takes a
cudaGraphicsResource_t, it is necessary to explicitly cast the CUgraphicsResource to a
cudaGraphicsResource_t.

In order to use a cudaGraphicsResource_t in a CUDA Driver API function which takes a
CUgraphicsResource, it is necessary to explicitly cast the cudaGraphicsResource_t to a
CUgraphicsResource.

Interactions between CUtexObject and cudaTextureObject_t

The types CUtexObject and cudaTextureObject_t represent the same data type and may be used
interchangeably by casting the two types between each other.

In order to use a CUtexObject in a CUDA Runtime API function which takes a cudaTextureObject_t, it
is necessary to explicitly cast the CUtexObject to a cudaTextureObject_t.

In order to use a cudaTextureObject_t in a CUDA Driver API function which takes a CUtexObject, it is
necessary to explicitly cast the cudaTextureObject_t to a CUtexObject.


CUDA Runtime API vRelease Version  |  517


Modules


Interactions between CUsurfObject and cudaSurfaceObject_t

The types CUsurfObject and cudaSurfaceObject_t represent the same data type and may be used
interchangeably by casting the two types between each other.

In order to use a CUsurfObject in a CUDA Runtime API function which takes a cudaSurfaceObject_t,
it is necessary to explicitly cast the CUsurfObject to a cudaSurfaceObject_t.

In order to use a cudaSurfaceObject_t in a CUDA Driver API function which takes a CUsurfObject, it
is necessary to explicitly cast the cudaSurfaceObject_t to a CUsurfObject.

Interactions between CUfunction and cudaFunction_t

The types CUfunction and cudaFunction_t represent the same data type and may be used
interchangeably by casting the two types between each other.

In order to use a cudaFunction_t in a CUDA Driver API function which takes a CUfunction, it is
necessary to explicitly cast the cudaFunction_t to a CUfunction.

Interactions between CUkernel and cudaKernel_t

The types CUkernel and cudaKernel_t represent the same data type and may be used interchangeably
by casting the two types between each other.

In order to use a cudaKernel_t in a CUDA Driver API function which takes a CUkernel, it is necessary
to explicitly cast the cudaKernel_t to a CUkernel.