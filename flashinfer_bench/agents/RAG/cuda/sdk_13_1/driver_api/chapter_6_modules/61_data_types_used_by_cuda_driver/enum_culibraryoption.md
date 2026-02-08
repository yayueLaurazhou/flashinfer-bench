# enum CUlibraryOption

Library options to be specified with cuLibraryLoadData() or cuLibraryLoadFromFile()


CUDA Driver API TRM-06703-001 _vRelease Version  |  61


Modules

###### Values

**CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = 0**
**CU_LIBRARY_BINARY_IS_PRESERVED = 1**
Specifes that the argument code passed to cuLibraryLoadData() will be preserved. Specifying this
option will let the driver know that code can be accessed at any point until cuLibraryUnload().
The default behavior is for the driver to allocate and maintain its own copy of code. Note
that this is only a memory usage optimization hint and the driver can choose to ignore it if
required. Specifying this option with cuLibraryLoadFromFile() is invalid and will return
CUDA_ERROR_INVALID_VALUE.
**CU_LIBRARY_NUM_OPTIONS**