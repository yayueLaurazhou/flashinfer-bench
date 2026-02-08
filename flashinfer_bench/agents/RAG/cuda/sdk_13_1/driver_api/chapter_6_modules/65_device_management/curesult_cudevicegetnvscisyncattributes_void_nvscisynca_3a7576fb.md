# CUresult cuDeviceGetNvSciSyncAttributes (void *nvSciSyncAttrList, CUdevice dev, int flags)

Return NvSciSync attributes that this device can support.

###### Parameters

**nvSciSyncAttrList**

  - Return NvSciSync attributes supported.
**dev**

  - Valid Cuda Device to get NvSciSync attributes for.
**flags**

  - flags describing NvSciSync usage.

###### Description

Returns in nvSciSyncAttrList, the properties of NvSciSync that this CUDA device, dev
can support. The returned nvSciSyncAttrList can be used to create an NvSciSync object that
matches this device's capabilities.

If NvSciSyncAttrKey_RequiredPerm field in nvSciSyncAttrList is already set this API will
return CUDA_ERROR_INVALID_VALUE.

The applications should set nvSciSyncAttrList to a valid NvSciSyncAttrList failing which this
API will return CUDA_ERROR_INVALID_HANDLE.

The flags controls how applications intends to use the NvSciSync created from the
nvSciSyncAttrList. The valid flags are:

CUDA_NVSCISYNC_ATTR_SIGNAL, specifies that the applications intends to signal an

###### **‣**

NvSciSync on this CUDA device.
CUDA_NVSCISYNC_ATTR_WAIT, specifies that the applications intends to wait on an

###### **‣**

NvSciSync on this CUDA device.

At least one of these flags must be set, failing which the API returns
CUDA_ERROR_INVALID_VALUE. Both the flags are orthogonal to one another: a developer
may set both these flags that allows to set both wait and signal specific attributes in the same
nvSciSyncAttrList.


CUDA Driver API TRM-06703-001 _vRelease Version  |  107


Modules


Note that this API updates the input nvSciSyncAttrList with values equivalent to the following
public attribute key-values: NvSciSyncAttrKey_RequiredPerm is set to

NvSciSyncAccessPerm_SignalOnly if CUDA_NVSCISYNC_ATTR_SIGNAL is set in .

###### ‣ flags

NvSciSyncAccessPerm_WaitOnly if CUDA_NVSCISYNC_ATTR_WAIT is set in .

###### ‣ flags

NvSciSyncAccessPerm_WaitSignal if both CUDA_NVSCISYNC_ATTR_WAIT and

###### **‣**

CUDA_NVSCISYNC_ATTR_SIGNAL are set in flags. NvSciSyncAttrKey_PrimitiveInfo is set
to
NvSciSyncAttrValPrimitiveType_SysmemSemaphore on any valid .

###### ‣ device

NvSciSyncAttrValPrimitiveType_Syncpoint if is a Tegra device.

###### ‣ device

NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b if is GA10X+.

###### ‣ device

NvSciSyncAttrKey_GpuId is set to the same UUID that is returned for this device from
cuDeviceGetUuid.


CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_SUPPORTED,
CUDA_ERROR_OUT_OF_MEMORY

See also:

cuImportExternalSemaphore, cuDestroyExternalSemaphore, cuSignalExternalSemaphoresAsync,
cuWaitExternalSemaphoresAsync