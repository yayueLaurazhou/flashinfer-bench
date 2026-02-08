# void *cudaMemPoolProps::win32SecurityAttributes

Windows-specific LPSECURITYATTRIBUTES required when cudaMemHandleTypeWin32 is
specified. This security attribute defines the scope of which exported allocations may be tranferred to
other processes. In all other cases, this field is required to be zero.