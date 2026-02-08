# __device__ void *cudaGetParameterBuffer (size_t alignment, size_t size)

Obtains a parameter buffer.

##### Parameters

**alignment**

  - Specifies alignment requirement of the parameter buffer
**size**

  - Specifies size requirement in bytes

##### Returns

Returns pointer to the allocated parameterBuffer

##### Description

Obtains a parameter buffer which can be filled with parameters for a kernel launch. Parameters passed
to cudaLaunchDevice must be allocated via this function.

This is a low level API and can only be accessed from Parallel Thread Execution (PTX). CUDA user
code should use <<< >>> to launch kernels.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaLaunchDevice


CUDA Runtime API vRelease Version  |  101


Modules