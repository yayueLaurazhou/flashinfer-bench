# Context management

Unless an execution context cudaExecutionContext_t is specified, the runtime API decides itself which
context to use for a thread: if a context has been made current to the calling thread through the driver
API, the runtime will use that, but if there is no such context, it uses the device execution context
which is a "primary context." Primary contexts are created as needed, one per device per process, are
reference-counted, and are then destroyed when there are no more references to them. Within one
process, all users of the runtime API will share the primary context, unless a context has been made
current to each thread or an explicit execution context is specified to the runtime APIs.

Using the runtime API with primary contexts has its tradeoffs, however. It can cause trouble for users
writing plug-ins for larger software packages, for example, because if all plug-ins run in the same
process, they will all share a context but will likely have no way to communicate with each other. So,
if one of them calls cudaDeviceReset() after finishing all its CUDA work, the other plug-ins will
fail because the context they were using was destroyed without their knowledge. To avoid this issue,
CUDA clients can use the driver API to create and set the current context, and then use the runtime API
to work with it. However, contexts may consume significant resources, such as device memory, extra
host threads, and performance costs of context switching on the device. This runtime-driver context


CUDA Driver API TRM-06703-001 _vRelease Version  |  1


Difference between the driver and runtime APIs


sharing is important when using the driver API in conjunction with libraries built on the runtime API,
such as cuBLAS or cuFFT.


CUDA Driver API TRM-06703-001 _vRelease Version  |  2