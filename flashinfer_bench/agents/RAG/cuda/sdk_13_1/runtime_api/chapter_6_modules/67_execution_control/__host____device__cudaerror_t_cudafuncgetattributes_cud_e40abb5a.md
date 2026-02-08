# __host____device__cudaError_t cudaFuncGetAttributes (cudaFuncAttributes *attr, const void *func)

Find out attributes for a given function.

##### Parameters

**attr**

  - Return pointer to function's attributes
**func**

  - Device function symbol

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction

##### Description

This function obtains the attributes of a function specified via func. func is a device function symbol
and must be declared as a __global__ function. The fetched attributes are placed in attr. If the
specified function does not exist, then it is assumed to be a cudaKernel_t and used as is. For templated
functions, pass the function symbol as follows: func_name<template_arg_0,...,template_arg_N>


CUDA Runtime API vRelease Version  |  95


Modules


Note that some function attributes such as maxThreadsPerBlock may vary based on the device that is
currently being used.









See also:

cudaFuncSetCacheConfig ( C API), cudaFuncGetAttributes ( C++ API), cudaLaunchKernel ( C API),
cuFuncGetAttribute