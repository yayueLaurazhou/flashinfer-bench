# enum cudaMemAllocationHandleType

Flags for specifying particular handle types

##### Values

**cudaMemHandleTypeNone = 0x0**
Does not allow any export mechanism. >
**cudaMemHandleTypePosixFileDescriptor = 0x1**
Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)
**cudaMemHandleTypeWin32 = 0x2**
Allows a Win32 NT handle to be used for exporting. (HANDLE)
**cudaMemHandleTypeWin32Kmt = 0x4**
Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
**cudaMemHandleTypeFabric = 0x8**
Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t)