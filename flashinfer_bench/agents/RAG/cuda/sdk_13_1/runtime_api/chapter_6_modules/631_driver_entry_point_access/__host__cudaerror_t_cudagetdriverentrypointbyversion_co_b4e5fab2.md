# __host__cudaError_t cudaGetDriverEntryPointByVersion (const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult *driverStatus)

Returns the requested driver API function pointer by CUDA version.

##### Parameters

**symbol**

  - The base name of the driver API function to look for. As an example, for the driver API
cuMemAlloc_v2, symbol would be cuMemAlloc.
**funcPtr**

  - Location to return the function pointer to the requested driver function
**cudaVersion**

  - The CUDA version to look for the requested driver symbol
**flags**

  - Flags to specify search options.
**driverStatus**

  - Optional location to store the status of finding the symbol from the driver. See
cudaDriverEntryPointQueryResult for possible values.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported

##### Description

Returns in **funcPtr the address of the CUDA driver function for the requested flags and CUDA
driver version.

The CUDA version is specified as (1000 * major + 10 * minor), so CUDA 11.2 should be specified
as 11020. For a requested driver symbol, if the specified CUDA version is greater than or equal to the
CUDA version in which the driver symbol was introduced, this API will return the function pointer to


CUDA Runtime API vRelease Version  |  430


Modules


the corresponding versioned function. If the specified CUDA version is greater than the driver version,
the API will return cudaErrorInvalidValue.

The pointer returned by the API should be cast to a function pointer matching the requested driver
function's definition in the API header file. The function pointer typedef can be picked up from the
corresponding typedefs header file. For example, cudaTypedefs.h consists of function pointer typedefs
for driver APIs defined in cuda.h.

For the case where the CUDA version requested is greater than the CUDA Toolkit installed, there may
not be an appropriate function pointer typedef in the corresponding header file and may need a custom
typedef to match the driver function signature returned. This can be done by getting the typedefs from a
later toolkit or creating appropriately matching custom function typedefs.

The API will return cudaSuccess and set the returned funcPtr if the requested driver function is valid
and supported on the platform.

The API will return cudaSuccess and set the returned funcPtr to NULL if the requested driver
function is not supported on the platform, no ABI compatible driver function exists for the requested
version or if the driver symbol is invalid.

It will also set the optional driverStatus to one of the values in cudaDriverEntryPointQueryResult
with the following meanings:

cudaDriverEntryPointSuccess   - The requested symbol was succesfully found based on input

##### **‣**

arguments and pfn is valid
cudaDriverEntryPointSymbolNotFound  - The requested symbol was not found

##### **‣**

cudaDriverEntryPointVersionNotSufficent   - The requested symbol was found but is not supported

##### **‣**

by the specified version cudaVersion

The requested flags can be:

cudaEnableDefault: This is the default mode. This is equivalent to

##### **‣**

cudaEnablePerThreadDefaultStream if the code is compiled with --default-stream per-thread
compilation flag or the macro CUDA_API_PER_THREAD_DEFAULT_STREAM is defined;
cudaEnableLegacyStream otherwise.
cudaEnableLegacyStream: This will enable the search for all driver symbols that match the

##### **‣**

requested driver symbol name except the corresponding per-thread versions.
cudaEnablePerThreadDefaultStream: This will enable the search for all driver symbols that match

##### **‣**

the requested driver symbol name including the per-thread versions. If a per-thread version is not
found, the API will return the legacy version of the driver function.


CUDA Runtime API vRelease Version  |  431


Modules





See also:

cuGetProcAddress