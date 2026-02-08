# void *cudaPointerAttributes::hostPointer

The address which may be dereferenced on the host to access the memory or NULL if no such address
exists.


Note:


CUDA doesn't check if unregistered memory is allocated so this field may contain invalid pointer if an
invalid pointer has been passed to CUDA.