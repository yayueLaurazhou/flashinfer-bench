# template < class T > __host__cudaError_t cudaFuncGetAttributes (cudaFuncAttributes *attr, T *entry)

[C++ API] Find out attributes for a given function

##### Parameters

**attr**

  - Return pointer to function's attributes
**entry**

  - Function to get attributes of

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction

##### Description

This function obtains the attributes of a function specified via entry. The parameter entry must be a
pointer to a function that executes on the device. The parameter specified by entry must be declared
as a __global__ function. The fetched attributes are placed in attr. If the specified function does
not exist, then cudaErrorInvalidDeviceFunction is returned.

Note that some function attributes such as maxThreadsPerBlock may vary based on the device that is
currently being used.


CUDA Runtime API vRelease Version  |  464


Modules


cudaLaunchKernel ( C++ API), cudaFuncSetCacheConfig ( C++ API), cudaFuncGetAttributes ( C
API), cudaSetDoubleForDevice, cudaSetDoubleForHost