# enum cudaMemAllocationType

Defines the allocation types available

##### Values

**cudaMemAllocationTypeInvalid = 0x0**
**cudaMemAllocationTypePinned = 0x1**
This allocation type is 'pinned', i.e. cannot migrate from its current location while the application is
actively using it
**cudaMemAllocationTypeManaged = 0x2**
This allocation type is managed memory
**cudaMemAllocationTypeMax = 0x7FFFFFFF**


CUDA Runtime API vRelease Version  |  570


Modules