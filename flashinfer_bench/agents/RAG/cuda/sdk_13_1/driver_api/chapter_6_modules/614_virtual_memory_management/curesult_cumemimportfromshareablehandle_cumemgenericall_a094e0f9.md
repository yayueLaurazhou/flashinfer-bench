# CUresult cuMemImportFromShareableHandle (CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType)

Imports an allocation from a requested shareable handle type.

###### Parameters

**handle**

  - CUDA Memory handle for the memory allocation.
**osHandle**

  - Shareable Handle representing the memory allocation that is to be imported.
**shHandleType**

  - handle type of the exported handle CUmemAllocationHandleType.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED


CUDA Driver API TRM-06703-001 _vRelease Version  |  277


Modules

###### Description

If the current process cannot support the memory described by this shareable handle, this API will error
as CUDA_ERROR_NOT_SUPPORTED.

If shHandleType is CU_MEM_HANDLE_TYPE_FABRIC and the importer process has not
been granted access to the same IMEX channel as the exporter process, this API will error as
CUDA_ERROR_NOT_PERMITTED.


Note:


Importing shareable handles exported from some graphics APIs(VUlkan, OpenGL, etc)
created on devices under an SLI group may not be supported, and thus this API will return
CUDA_ERROR_NOT_SUPPORTED. There is no guarantee that the contents of handle will be the
same CUDA memory handle for the same given OS shareable handle, or the same underlying allocation.


See also:

cuMemExportToShareableHandle, cuMemMap, cuMemRelease