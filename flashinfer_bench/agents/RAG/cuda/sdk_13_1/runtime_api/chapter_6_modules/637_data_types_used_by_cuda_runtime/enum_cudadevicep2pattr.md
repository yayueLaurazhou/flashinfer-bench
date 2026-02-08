# enum cudaDeviceP2PAttr

CUDA device P2P attributes

##### Values

**cudaDevP2PAttrPerformanceRank = 1**
A relative value indicating the performance of the link between two devices
**cudaDevP2PAttrAccessSupported = 2**
Peer access is enabled
**cudaDevP2PAttrNativeAtomicSupported = 3**
Native atomic operation over the link supported
**cudaDevP2PAttrCudaArrayAccessSupported = 4**
Accessing CUDA arrays over the link supported
**cudaDevP2PAttrOnlyPartialNativeAtomicSupported = 5**
Only some CUDA-valid atomic operations over the link are supported.