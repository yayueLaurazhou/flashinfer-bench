# #define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_L 0x01

If set, each kernel launched as part of cuLaunchCooperativeKernelMultiDevice only waits for prior
work in the stream corresponding to that GPU to complete before the kernel begins execution.