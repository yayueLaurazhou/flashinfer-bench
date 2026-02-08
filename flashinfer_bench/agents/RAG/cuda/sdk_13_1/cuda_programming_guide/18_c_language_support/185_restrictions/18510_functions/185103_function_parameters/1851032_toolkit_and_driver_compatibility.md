# 18.5.10.3.2. Toolkit and Driver Compatibility

##### 18.5.10.3.2. Toolkit and Driver Compatibility[](#toolkit-and-driver-compatibility "Permalink to this headline")

Developers must use the 12.1 Toolkit and r530 driver or higher to compile, launch, and debug kernels that accept parameters larger than 4KB. If such kernels are launched on older drivers, CUDA will issue the error `CUDA_ERROR_NOT_SUPPORTED`.