# void *CUmemAllocationProp_v1::win32HandleMetaData

Windows-specific POBJECT_ATTRIBUTES required when CU_MEM_HANDLE_TYPE_WIN32
is specified. This object attributes structure includes security attributes that define the scope of which
exported allocations may be transferred to other processes. In all other cases, this field is required to be
zero.