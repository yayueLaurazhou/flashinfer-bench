# #define CUDA_NVSCISYNC_ATTR_WAIT 0x2

When flags of cuDeviceGetNvSciSyncAttributes is set to this, it indicates that application needs
waiter specific NvSciSyncAttr to be filled by cuDeviceGetNvSciSyncAttributes.