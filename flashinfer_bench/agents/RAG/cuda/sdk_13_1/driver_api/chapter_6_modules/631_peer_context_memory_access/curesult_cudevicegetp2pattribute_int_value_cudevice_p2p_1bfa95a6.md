# CUresult cuDeviceGetP2PAttribute (int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice)

Queries attributes of the link between two devices.

###### Parameters

**value**

  - Returned value of the requested attribute
**attrib**

  - The requested attribute of the link between srcDevice and dstDevice.
**srcDevice**

  - The source device of the target link.
**dstDevice**

  - The destination device of the target link.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in *value the value of the requested attribute attrib of the link between srcDevice and
dstDevice. The supported attributes are:

CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: A relative value indicating the

###### **‣**

performance of the link between two devices.
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED P2P: 1 if P2P Access is enable.

###### **‣**

CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED: 1 if all CUDA-valid

###### **‣**

atomic operations over the link are supported.
CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED: 1 if cudaArray can

###### **‣**

be accessed over the link.
CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED: 1

###### **‣**

if some CUDA-valid atomic operations over the link are supported. Information about specific
operations can be retrieved with cuDeviceGetP2PAtomicCapabilities.

Returns CUDA_ERROR_INVALID_DEVICE if srcDevice or dstDevice are not valid or if they
represent the same device.

Returns CUDA_ERROR_INVALID_VALUE if attrib is not valid or if value is a null pointer.


CUDA Driver API TRM-06703-001 _vRelease Version  |  555


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxEnablePeerAccess, cuCtxDisablePeerAccess, cuDeviceCanAccessPeer,
cuDeviceGetP2PAtomicCapabilities, cudaDeviceGetP2PAttribute