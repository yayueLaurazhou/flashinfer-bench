# enum cudaEglResourceLocationFlags

Resource location flags- sysmem or vidmem

For CUDA context on iGPU, since video and system memory are equivalent - these flags will not have
an effect on the execution.

For CUDA context on dGPU, applications can use the flag cudaEglResourceLocationFlags to give a
hint about the desired location.

cudaEglResourceLocationSysmem - the frame data is made resident on the system memory to be
accessed by CUDA.

cudaEglResourceLocationVidmem - the frame data is made resident on the dedicated video memory to
be accessed by CUDA.

There may be an additional latency due to new allocation and data migration, if the frame is produced
on a different memory.


CUDA Runtime API vRelease Version  |  543


Modules

##### Values

**cudaEglResourceLocationSysmem = 0x00**
Resource location sysmem
**cudaEglResourceLocationVidmem = 0x01**
Resource location vidmem