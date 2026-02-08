# enum CUdevice_P2PAttribute

P2P Attributes

###### Values

**CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01**
A relative value indicating the performance of the link between two devices
**CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02**
P2P Access is enable
**CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03**
All CUDA-valid atomic operation over the link are supported
**CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 0x04**

Deprecated use CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED
instead
**CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 0x04**
Accessing CUDA arrays over the link supported
**CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED =**
**0x05**
Only some CUDA-valid atomic operations over the link are supported.


CUDA Driver API TRM-06703-001 _vRelease Version  |  31


Modules