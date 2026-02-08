# __host__cudaError_t cudaPointerGetAttributes (cudaPointerAttributes *attributes, const void *ptr)

Returns attributes about a specified pointer.

##### Parameters

**attributes**

  - Attributes for the specified pointer
**ptr**

  - Pointer to get attributes for

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

##### Description

Returns in *attributes the attributes of the pointer ptr. If pointer was not allocated in, mapped by
or registered with context supporting unified addressing cudaErrorInvalidValue is returned.





The cudaPointerAttributes structure is defined as:


CUDA Runtime API vRelease Version  |  227


Modules


}
In this structure, the individual fields mean

cudaPointerAttributes::type identifies type of memory. It can be cudaMemoryTypeUnregistered

##### **‣**

for unregistered host memory, cudaMemoryTypeHost for registered host memory,
cudaMemoryTypeDevice for device memory or cudaMemoryTypeManaged for managed memory.

device is the device against which was allocated. If has memory type

##### ‣ ptr ptr

cudaMemoryTypeDevice then this identifies the device on which the memory referred to by ptr
physically resides. If ptr has memory type cudaMemoryTypeHost then this identifies the device
which was current when the allocation was made (and if that device is deinitialized then this
allocation will vanish with that device's state).

devicePointer is the device pointer alias through which the memory referred to by may be

##### ‣ ptr

accessed on the current device. If the memory referred to by ptr cannot be accessed directly by
the current device then this is NULL.

hostPointer is the host pointer alias through which the memory referred to by may be accessed

##### ‣ ptr

on the host. If the memory referred to by ptr cannot be accessed directly by the host then this is
NULL.





See also:

cudaGetDeviceCount, cudaGetDevice, cudaSetDevice, cudaChooseDevice, cudaInitDevice,
cuPointerGetAttributes