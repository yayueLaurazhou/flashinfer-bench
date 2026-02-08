# enum CUeglResourceLocationFlags

Resource location flags- sysmem or vidmem

For CUDA context on iGPU, since video and system memory are equivalent - these flags will not have
an effect on the execution.

For CUDA context on dGPU, applications can use the flag CUeglResourceLocationFlags to give a hint
about the desired location.

CU_EGL_RESOURCE_LOCATION_SYSMEM - the frame data is made resident on the system
memory to be accessed by CUDA.

CU_EGL_RESOURCE_LOCATION_VIDMEM - the frame data is made resident on the dedicated
video memory to be accessed by CUDA.

There may be an additional latency due to new allocation and data migration, if the frame is produced
on a different memory.

###### Values

**CU_EGL_RESOURCE_LOCATION_SYSMEM = 0x00**
Resource location sysmem
**CU_EGL_RESOURCE_LOCATION_VIDMEM = 0x01**
Resource location vidmem