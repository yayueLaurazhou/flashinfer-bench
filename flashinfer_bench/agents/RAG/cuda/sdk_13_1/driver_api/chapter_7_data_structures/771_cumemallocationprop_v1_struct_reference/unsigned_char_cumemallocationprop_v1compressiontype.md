# unsigned char CUmemAllocationProp_v1::compressionType

Allocation hint for requesting compressible memory. On devices that support Compute Data
Compression, compressible memory can be used to accelerate accesses to data with unstructured
sparsity and other compressible data patterns. Applications are expected to query allocation property of
the handle obtained with cuMemCreate using cuMemGetAllocationPropertiesFromHandle to validate
if the obtained allocation is compressible or not. Note that compressed memory may not be mappable
on all devices.