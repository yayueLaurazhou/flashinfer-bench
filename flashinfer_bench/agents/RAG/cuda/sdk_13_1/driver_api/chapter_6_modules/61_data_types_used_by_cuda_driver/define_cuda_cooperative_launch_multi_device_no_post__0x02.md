# #define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_ 0x02

If set, any subsequent work pushed in a stream that participated in a call to
cuLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on the GPU
corresponding to that stream to complete before it begins execution.