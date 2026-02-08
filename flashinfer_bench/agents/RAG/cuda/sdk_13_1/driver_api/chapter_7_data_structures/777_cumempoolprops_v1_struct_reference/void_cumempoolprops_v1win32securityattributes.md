# void *CUmemPoolProps_v1::win32SecurityAttributes

Windows-specific LPSECURITYATTRIBUTES required when CU_MEM_HANDLE_TYPE_WIN32
is specified. This security attribute defines the scope of which exported allocations may be transferred
to other processes. In all other cases, this field is required to be zero.