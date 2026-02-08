# struct CUlaunchConfig struct CUlaunchMemSyncDomainMap struct CUmemAccessDesc_v1 struct CUmemAllocationProp_v1 struct CUmemcpy3DOperand_v1 struct CUmemcpyAttributes_v1 struct CUmemFabricHandle_v1 struct CUmemLocation_v1 struct CUmemPoolProps_v1 struct CUmemPoolPtrExportData_v1 struct CUmulticastObjectProp_v1 struct CUoffset3D_v1 union CUstreamBatchMemOpParams_v1 struct CUtensorMap enum cl_context_flags

NVCL context scheduling flags

###### Values

**NVCL_CTX_SCHED_AUTO = 0x00**
Automatic scheduling


CUDA Driver API TRM-06703-001 _vRelease Version  |  15


Modules


**NVCL_CTX_SCHED_SPIN = 0x01**
Set spin as default scheduling
**NVCL_CTX_SCHED_YIELD = 0x02**
Set yield as default scheduling
**NVCL_CTX_SCHED_BLOCKING_SYNC = 0x04**
Set blocking synchronization as default scheduling