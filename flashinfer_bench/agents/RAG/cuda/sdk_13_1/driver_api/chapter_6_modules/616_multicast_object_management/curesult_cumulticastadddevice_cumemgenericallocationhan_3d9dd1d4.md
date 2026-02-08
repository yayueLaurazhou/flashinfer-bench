# CUresult cuMulticastAddDevice (CUmemGenericAllocationHandle mcHandle, CUdevice dev)

Associate a device to a multicast object.

###### Parameters

**mcHandle**
Handle representing a multicast object.
**dev**
Device that will be associated to the multicast object.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Associates a device to a multicast object. The added device will be a part of the multicast team of size
specified by CUmulticastObjectProp::numDevices during cuMulticastCreate. The association of the
device to the multicast object is permanent during the life time of the multicast object. All devices must
be added to the multicast team before any memory can be bound to any device in the team. Any calls to
cuMulticastBindMem, cuMulticastBindMem_v2, cuMulticastBindAddr, or cuMulticastBindAddr_v2
will block until all devices have been added. Similarly all devices must be added to the multicast team
before a virtual address range can be mapped to the multicast object. A call to cuMemMap will block
until all devices have been added.


See also:

cuMulticastCreate, cuMulticastBindMem, cuMulticastBindAddr


CUDA Driver API TRM-06703-001 _vRelease Version  |  302


Modules