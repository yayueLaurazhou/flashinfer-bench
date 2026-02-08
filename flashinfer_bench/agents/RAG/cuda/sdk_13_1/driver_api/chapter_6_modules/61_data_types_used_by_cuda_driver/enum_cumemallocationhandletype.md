# enum CUmemAllocationHandleType

Flags for specifying particular handle types

###### Values

**CU_MEM_HANDLE_TYPE_NONE = 0x0**
Does not allow any export mechanism. >
**CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1**
Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)
**CU_MEM_HANDLE_TYPE_WIN32 = 0x2**
Allows a Win32 NT handle to be used for exporting. (HANDLE)
**CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4**
Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
**CU_MEM_HANDLE_TYPE_FABRIC = 0x8**
Allows a fabric handle to be used for exporting. (CUmemFabricHandle)
**CU_MEM_HANDLE_TYPE_MAX = 0x7FFFFFFF**