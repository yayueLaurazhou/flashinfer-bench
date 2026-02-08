# enum cudaLibraryOption

Library options to be specified with cudaLibraryLoadData() or cudaLibraryLoadFromFile()

##### Values

**cudaLibraryHostUniversalFunctionAndDataTable = 0**
**cudaLibraryBinaryIsPreserved = 1**
Specifes that the argument code passed to cudaLibraryLoadData() will be preserved.
Specifying this option will let the driver know that code can be accessed at any point until
cudaLibraryUnload(). The default behavior is for the driver to allocate and maintain its own copy of
code. Note that this is only a memory usage optimization hint and the driver can choose to ignore
it if required. Specifying this option with cudaLibraryLoadFromFile() is invalid and will return
cudaErrorInvalidValue.