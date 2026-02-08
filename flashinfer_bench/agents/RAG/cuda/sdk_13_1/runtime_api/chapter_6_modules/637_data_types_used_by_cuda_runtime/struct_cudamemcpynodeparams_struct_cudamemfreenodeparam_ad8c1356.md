# struct cudaMemcpyNodeParams struct cudaMemFreeNodeParams struct cudaMemLocation struct cudaMemPoolProps struct cudaMemPoolPtrExportData struct cudaMemsetParams struct cudaMemsetParamsV2 struct cudaOffset3D struct cudaPitchedPtr struct cudaPointerAttributes struct cudaPos struct cudaResourceDesc struct cudaResourceViewDesc struct cudaTextureDesc struct CUuuid_st enum cudaAccessProperty

Specifies performance hint with cudaAccessPolicyWindow for hitProp and missProp members.


CUDA Runtime API vRelease Version  |  524


Modules

##### Values

**cudaAccessPropertyNormal = 0**
Normal cache persistence.
**cudaAccessPropertyStreaming = 1**
Streaming access is less likely to persit from cache.
**cudaAccessPropertyPersisting = 2**
Persisting access is more likely to persist in cache.