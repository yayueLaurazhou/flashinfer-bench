# __host__cudaError_t cudaDeviceGetNvSciSyncAttributes (void *nvSciSyncAttrList, int device, int flags)

Return NvSciSync attributes that this device can support.

##### Parameters

**nvSciSyncAttrList**

  - Return NvSciSync attributes supported.
**device**

  - Valid Cuda Device to get NvSciSync attributes for.
**flags**

  - flags describing NvSciSync usage.

##### Description

Returns in nvSciSyncAttrList, the properties of NvSciSync that this CUDA device, dev can
support. The returned nvSciSyncAttrList can be used to create an NvSciSync that matches this
device's capabilities.

If NvSciSyncAttrKey_RequiredPerm field in nvSciSyncAttrList is already set this API will
return cudaErrorInvalidValue.

The applications should set nvSciSyncAttrList to a valid NvSciSyncAttrList failing which this
API will return cudaErrorInvalidHandle.

The flags controls how applications intends to use the NvSciSync created from the
nvSciSyncAttrList. The valid flags are:

cudaNvSciSyncAttrSignal, specifies that the applications intends to signal an NvSciSync on this

##### **‣**

CUDA device.
cudaNvSciSyncAttrWait, specifies that the applications intends to wait on an NvSciSync on this

##### **‣**

CUDA device.

At least one of these flags must be set, failing which the API returns cudaErrorInvalidValue. Both the
flags are orthogonal to one another: a developer may set both these flags that allows to set both wait
and signal specific attributes in the same nvSciSyncAttrList.

Note that this API updates the input nvSciSyncAttrList with values equivalent to the following
public attribute key-values: NvSciSyncAttrKey_RequiredPerm is set to

NvSciSyncAccessPerm_SignalOnly if cudaNvSciSyncAttrSignal is set in .

##### ‣ flags

NvSciSyncAccessPerm_WaitOnly if cudaNvSciSyncAttrWait is set in .

##### ‣ flags

NvSciSyncAccessPerm_WaitSignal if both cudaNvSciSyncAttrWait and

##### **‣**

cudaNvSciSyncAttrSignal are set in flags. NvSciSyncAttrKey_PrimitiveInfo is set to
NvSciSyncAttrValPrimitiveType_SysmemSemaphore on any valid .

##### ‣ device

CUDA Runtime API vRelease Version  |  18


Modules


NvSciSyncAttrValPrimitiveType_Syncpoint if is a Tegra device.

##### ‣ device

NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b if is

##### ‣ device

GA10X+. NvSciSyncAttrKey_GpuId is set to the same UUID that is returned in
cudaDeviceProp.uuid from cudaDeviceGetProperties for this device.


cudaSuccess, cudaErrorDeviceUninitialized, cudaErrorInvalidValue, cudaErrorInvalidHandle,
cudaErrorInvalidDevice, cudaErrorNotSupported, cudaErrorMemoryAllocation

See also:

cudaImportExternalSemaphore, cudaDestroyExternalSemaphore,
cudaSignalExternalSemaphoresAsync, cudaWaitExternalSemaphoresAsync