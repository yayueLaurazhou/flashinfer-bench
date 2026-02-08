# #define CUDA_NVSCISYNC_ATTR_SIGNAL 0x1

When flags of cuDeviceGetNvSciSyncAttributes is set to this, it indicates that application needs
signaler specific NvSciSyncAttr to be filled by cuDeviceGetNvSciSyncAttributes.