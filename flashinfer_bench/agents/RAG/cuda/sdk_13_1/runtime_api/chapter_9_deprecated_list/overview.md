# Chapter 9. Deprecated List

**Global cudaDeviceGetSharedMemConfig**


**Global cudaDeviceSetSharedMemConfig**


**Global cudaFuncSetSharedMemConfig**


**Global cudaMemcpyArrayToArray**


**Global cudaMemcpyFromArray**


**Global cudaMemcpyFromArrayAsync**


**Global cudaMemcpyToArray**


**Global cudaMemcpyToArrayAsync**


**Global cudaGLMapBufferObject**

This function is deprecated as of CUDA 3.0.


**Global cudaGLMapBufferObjectAsync**

This function is deprecated as of CUDA 3.0.


CUDA Runtime API vRelease Version  |  666


Deprecated List


**Global cudaGLRegisterBufferObject**

This function is deprecated as of CUDA 3.0.


**Global cudaGLSetBufferObjectMapFlags**

This function is deprecated as of CUDA 3.0.


**Global cudaGLSetGLDevice**

This function is deprecated as of CUDA 5.0.


**Global cudaGLUnmapBufferObject**

This function is deprecated as of CUDA 3.0.


**Global cudaGLUnmapBufferObjectAsync**

This function is deprecated as of CUDA 3.0.


**Global cudaGLUnregisterBufferObject**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9MapResources**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9RegisterResource**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9ResourceGetMappedArray**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9ResourceGetMappedPitch**

This function is deprecated as of CUDA 3.0.


CUDA Runtime API vRelease Version  |  667


Deprecated List


**Global cudaD3D9ResourceGetMappedPointer**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9ResourceGetMappedSize**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9ResourceGetSurfaceDimensions**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9ResourceSetMapFlags**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9UnmapResources**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D9UnregisterResource**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10GetDirect3DDevice**

This function is deprecated as of CUDA 5.0.


**Global cudaD3D10MapResources**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10RegisterResource**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10ResourceGetMappedArray**

This function is deprecated as of CUDA 3.0.


CUDA Runtime API vRelease Version  |  668


Deprecated List


**Global cudaD3D10ResourceGetMappedPitch**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10ResourceGetMappedPointer**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10ResourceGetMappedSize**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10ResourceGetSurfaceDimensions**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10ResourceSetMapFlags**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10SetDirect3DDevice**

This function is deprecated as of CUDA 5.0.


**Global cudaD3D10UnmapResources**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D10UnregisterResource**

This function is deprecated as of CUDA 3.0.


**Global cudaD3D11GetDirect3DDevice**

This function is deprecated as of CUDA 5.0.


**Global cudaD3D11SetDirect3DDevice**

This function is deprecated as of CUDA 5.0.


CUDA Runtime API vRelease Version  |  669


Deprecated List


**Global cudaGetDriverEntryPoint**

This function is deprecated as of CUDA 13.0


**Global cudaErrorProfilerNotInitialized**

This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable
the profiling via cudaProfilerStart or cudaProfilerStop without initialization.


**Global cudaErrorProfilerAlreadyStarted**

This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStart()
when profiling is already enabled.


**Global cudaErrorProfilerAlreadyStopped**

This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStop()
when profiling is already disabled.


**Global cudaErrorInvalidHostPointer**

This error return is deprecated as of CUDA 10.1.


**Global cudaErrorInvalidDevicePointer**

This error return is deprecated as of CUDA 10.1.


**Global cudaErrorAddressOfConstant**

This error return is deprecated as of CUDA 3.1. Variables in constant memory may now have their
address taken by the runtime via cudaGetSymbolAddress().


**Global cudaErrorTextureFetchFailed**

This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the
CUDA 3.1 release.


**Global cudaErrorTextureNotBound**

This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the
CUDA 3.1 release.


CUDA Runtime API vRelease Version  |  670


Deprecated List


**Global cudaErrorSynchronizationError**

This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the
CUDA 3.1 release.


**Global cudaErrorMixedDeviceExecution**

This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the
CUDA 3.1 release.


**Global cudaErrorNotYetImplemented**

This error return is deprecated as of CUDA 4.1.


**Global cudaErrorMemoryValueTooLarge**

This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the
CUDA 3.1 release.


**Global cudaErrorPriorLaunchFailure**

This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the
CUDA 3.1 release.


**Global cudaSharedMemConfig**


**Global cudaDeviceBlockingSync**

This flag was deprecated as of CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync.


CUDA Runtime API vRelease Version  |  671


Notice

This document is provided for information purposes only and shall not be regarded as a warranty of a certain functionality, condition, or quality of a product. NVIDIA Corporation
(“NVIDIA”) makes no representations or warranties, expressed or implied, as to the accuracy or completeness of the information contained in this document and assumes no
responsibility for any errors contained herein. NVIDIA shall have no liability for the consequences or use of such information or for any infringement of patents or other rights of
third parties that may result from its use. This document is not a commitment to develop, release, or deliver any Material (defined below), code, or functionality.


NVIDIA reserves the right to make corrections, modifications, enhancements, improvements, and any other changes to this document, at any time without notice.


Customer should obtain the latest relevant information before placing orders and should verify that such information is current and complete.


NVIDIA products are sold subject to the NVIDIA standard terms and conditions of sale supplied at the time of order acknowledgement, unless otherwise agreed in an individual
sales agreement signed by authorized representatives of NVIDIA and customer (“Terms of Sale”). NVIDIA hereby expressly objects to applying any customer general terms and
conditions with regards to the purchase of the NVIDIA product referenced in this document. No contractual obligations are formed either directly or indirectly by this document.


OpenCL

OpenCL is a trademark of Apple Inc. used under license to the Khronos Group Inc.


Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be
trademarks of the respective companies with which they are associated.