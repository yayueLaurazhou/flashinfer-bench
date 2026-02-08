# __host__cudaError_t cudaDeviceGetP2PAttribute (int *value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)

Queries attributes of the link between two devices.

##### Parameters

**value**

  - Returned value of the requested attribute
**attr**
**srcDevice**

  - The source device of the target link.
**dstDevice**

  - The destination device of the target link.

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

##### Description

Returns in *value the value of the requested attribute attrib of the link between srcDevice and
dstDevice. The supported attributes are:

cudaDevP2PAttrPerformanceRank: A relative value indicating the performance of the link between

##### **‣**

two devices. Lower value means better performance (0 being the value used for most performant
link).
cudaDevP2PAttrAccessSupported: 1 if peer access is enabled.

##### **‣**

cudaDevP2PAttrNativeAtomicSupported: 1 if all native atomic operations over the link are

##### **‣**

supported.


CUDA Runtime API vRelease Version  |  20


Modules


cudaDevP2PAttrCudaArrayAccessSupported: 1 if accessing CUDA arrays over the link is

##### **‣**

supported.
cudaDevP2PAttrOnlyPartialNativeAtomicSupported: 1 if some CUDA-valid atomic operations

##### **‣**

over the link are supported. Information about specific operations can be retrieved with
cudaDeviceGetP2PAtomicCapabilities.

Returns cudaErrorInvalidDevice if srcDevice or dstDevice are not valid or if they represent the
same device.

Returns cudaErrorInvalidValue if attrib is not valid or if value is a null pointer.



See also:

cudaDeviceEnablePeerAccess, cudaDeviceDisablePeerAccess, cudaDeviceCanAccessPeer,
cuDeviceGetP2PAttribute cudaDeviceGetP2PAtomicCapabilities