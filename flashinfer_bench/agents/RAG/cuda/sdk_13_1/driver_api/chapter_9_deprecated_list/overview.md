# Chapter 9. Deprecated List

**Global CU_CTX_BLOCKING_SYNC**

This flag was deprecated as of CUDA 4.0 and was replaced with
CU_CTX_SCHED_BLOCKING_SYNC.


**Global CU_CTX_MAP_HOST**

This flag was deprecated as of CUDA 11.0 and it no longer has any effect. All contexts as of CUDA
3.2 behave as though the flag is enabled.


**Global CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED**

use CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED instead


**Global CU_JIT_NEW_SM3X_OPT**

This jit option is deprecated and should not be used.


**Global CU_JIT_LTO**

Enable link-time optimization (-dlto) for device code (Disabled by default).

This option is not supported on 32-bit platforms.

Option type: int

Applies to: compiler and linker


**Global CU_JIT_FTZ**

Control single-precision denormals (-ftz) support (0: false, default). 1 : flushes denormal values to
zero 0 : preserves denormal values Option type: int

Applies to: link-time optimization specified with CU_JIT_LTO


CUDA Driver API TRM-06703-001 _vRelease Version  |  754


Deprecated List


**Global CU_JIT_PREC_DIV**

Control single-precision floating-point division and reciprocals (-prec-div) support (1: true, default).
1 : Enables the IEEE round-to-nearest mode 0 : Enables the fast approximation mode Option type:
int

Applies to: link-time optimization specified with CU_JIT_LTO


**Global CU_JIT_PREC_SQRT**

Control single-precision floating-point square root (-prec-sqrt) support (1: true, default). 1 : Enables
the IEEE round-to-nearest mode 0 : Enables the fast approximation mode Option type: int

Applies to: link-time optimization specified with CU_JIT_LTO


**Global CU_JIT_FMA**

Enable/Disable the contraction of floating-point multiplies and adds/subtracts into floating-point
multiply-add (-fma) operations (1: Enable, default; 0: Disable). Option type: int

Applies to: link-time optimization specified with CU_JIT_LTO


**Global CU_JIT_REFERENCED_KERNEL_NAMES**

Array of kernel names that should be preserved at link time while others can be removed.

Must contain CU_JIT_REFERENCED_KERNEL_COUNT entries.

Note that kernel names can be mangled by the compiler in which case the mangled name needs to
be specified.

Wildcard "*" can be used to represent zero or more characters instead of specifying the full or
mangled name.

It is important to note that the wildcard "*" is also added implicitly. For example, specifying "foo"
will match "foobaz", "barfoo", "barfoobaz" and thus preserve all kernels with those names. This can
be avoided by providing a more specific name like "barfoobaz".

Option type: const char **

Applies to: dynamic linker only


**Global CU_JIT_REFERENCED_KERNEL_COUNT**

Number of entries in CU_JIT_REFERENCED_KERNEL_NAMES array.

Option type: unsigned int

Applies to: dynamic linker only


CUDA Driver API TRM-06703-001 _vRelease Version  |  755


Deprecated List


**Global CU_JIT_REFERENCED_VARIABLE_NAMES**

Array of variable names (__device__ and/or __constant__) that should be preserved at link time
while others can be removed.

Must contain CU_JIT_REFERENCED_VARIABLE_COUNT entries.

Note that variable names can be mangled by the compiler in which case the mangled name needs to
be specified.

Wildcard "*" can be used to represent zero or more characters instead of specifying the full or
mangled name.

It is important to note that the wildcard "*" is also added implicitly. For example, specifying "foo"
will match "foobaz", "barfoo", "barfoobaz" and thus preserve all variables with those names. This
can be avoided by providing a more specific name like "barfoobaz".

Option type: const char **

Applies to: link-time optimization specified with CU_JIT_LTO


**Global CU_JIT_REFERENCED_VARIABLE_COUNT**

Number of entries in CU_JIT_REFERENCED_VARIABLE_NAMES array.

Option type: unsigned int

Applies to: link-time optimization specified with CU_JIT_LTO


**Global CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES**

This option serves as a hint to enable the JIT compiler/linker to remove constant (__constant__) and
device (__device__) variables unreferenced in device code (Disabled by default).

Note that host references to constant and device variables using APIs like cuModuleGetGlobal()
with this option specified may result in undefined behavior unless the variables are explicitly
specified using CU_JIT_REFERENCED_VARIABLE_NAMES.

Option type: int

Applies to: link-time optimization specified with CU_JIT_LTO


**Global CU_JIT_INPUT_NVVM**

High-level intermediate code for link-time optimization

Applicable options: NVVM compiler options, PTX compiler options


CUDA Driver API TRM-06703-001 _vRelease Version  |  756


Deprecated List


**Global CUDA_ERROR_PROFILER_NOT_INITIALIZED**

This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable
the profiling via cuProfilerStart or cuProfilerStop without initialization.


**Global CUDA_ERROR_PROFILER_ALREADY_STARTED**

This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStart() when
profiling is already enabled.


**Global CUDA_ERROR_PROFILER_ALREADY_STOPPED**

This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStop() when
profiling is already disabled.


**Global CUDA_ERROR_CONTEXT_ALREADY_CURRENT**

This error return is deprecated as of CUDA 3.2. It is no longer an error to attempt to push the active
context via cuCtxPushCurrent().


**Global CUsharedconfig**


**Global cuDeviceComputeCapability**


**Global cuDeviceGetProperties**


**Global cuCtxAttach**


**Global cuCtxDetach**


**Global cuCtxGetSharedMemConfig**


**Global cuCtxSetSharedMemConfig**


**Global cuModuleGetSurfRef**


CUDA Driver API TRM-06703-001 _vRelease Version  |  757


Deprecated List


**Global cuModuleGetTexRef**


**Global cuLaunchCooperativeKernelMultiDevice**

This function is deprecated as of CUDA 11.3.


**Global cuFuncSetBlockShape**


**Global cuFuncSetSharedMemConfig**


**Global cuFuncSetSharedSize**


**Global cuLaunch**


**Global cuLaunchGrid**


**Global cuLaunchGridAsync**


**Global cuParamSetf**


**Global cuParamSeti**


**Global cuParamSetSize**


**Global cuParamSetTexRef**


**Global cuParamSetv**


CUDA Driver API TRM-06703-001 _vRelease Version  |  758


Deprecated List


**Global cuTexRefCreate**


**Global cuTexRefDestroy**


**Global cuTexRefGetAddress**


**Global cuTexRefGetAddressMode**


**Global cuTexRefGetArray**


**Global cuTexRefGetBorderColor**


**Global cuTexRefGetFilterMode**


**Global cuTexRefGetFlags**


**Global cuTexRefGetFormat**


**Global cuTexRefGetMaxAnisotropy**


**Global cuTexRefGetMipmapFilterMode**


**Global cuTexRefGetMipmapLevelBias**


**Global cuTexRefGetMipmapLevelClamp**


**Global cuTexRefGetMipmappedArray**


CUDA Driver API TRM-06703-001 _vRelease Version  |  759


Deprecated List


**Global cuTexRefSetAddress**


**Global cuTexRefSetAddress2D**


**Global cuTexRefSetAddressMode**


**Global cuTexRefSetArray**


**Global cuTexRefSetBorderColor**


**Global cuTexRefSetFilterMode**


**Global cuTexRefSetFlags**


**Global cuTexRefSetFormat**


**Global cuTexRefSetMaxAnisotropy**


**Global cuTexRefSetMipmapFilterMode**


**Global cuTexRefSetMipmapLevelBias**


**Global cuTexRefSetMipmapLevelClamp**


**Global cuTexRefSetMipmappedArray**


CUDA Driver API TRM-06703-001 _vRelease Version  |  760


Deprecated List


**Global cuSurfRefGetArray**


**Global cuSurfRefSetArray**


**Global cuProfilerInitialize**


**Global cuGLCtxCreate**

This function is deprecated as of Cuda 5.0.


**Global cuGLInit**

This function is deprecated as of Cuda 3.0.


**Global cuGLMapBufferObject**

This function is deprecated as of Cuda 3.0.


**Global cuGLMapBufferObjectAsync**

This function is deprecated as of Cuda 3.0.


**Global cuGLRegisterBufferObject**

This function is deprecated as of Cuda 3.0.


**Global cuGLSetBufferObjectMapFlags**

This function is deprecated as of Cuda 3.0.


**Global cuGLUnmapBufferObject**

This function is deprecated as of Cuda 3.0.


**Global cuGLUnmapBufferObjectAsync**

This function is deprecated as of Cuda 3.0.


CUDA Driver API TRM-06703-001 _vRelease Version  |  761


Deprecated List


**Global cuGLUnregisterBufferObject**

This function is deprecated as of Cuda 3.0.


**Global cuD3D9MapResources**

This function is deprecated as of CUDA 3.0.


**Global cuD3D9RegisterResource**

This function is deprecated as of CUDA 3.0.


**Global cuD3D9ResourceGetMappedArray**

This function is deprecated as of CUDA 3.0.


**Global cuD3D9ResourceGetMappedPitch**

This function is deprecated as of CUDA 3.0.


**Global cuD3D9ResourceGetMappedPointer**

This function is deprecated as of CUDA 3.0.


**Global cuD3D9ResourceGetMappedSize**

This function is deprecated as of CUDA 3.0.


**Global cuD3D9ResourceGetSurfaceDimensions**

This function is deprecated as of CUDA 3.0.


**Global cuD3D9ResourceSetMapFlags**

This function is deprecated as of Cuda 3.0.


**Global cuD3D9UnmapResources**

This function is deprecated as of CUDA 3.0.


CUDA Driver API TRM-06703-001 _vRelease Version  |  762


Deprecated List


**Global cuD3D9UnregisterResource**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10CtxCreate**

This function is deprecated as of CUDA 5.0.


**Global cuD3D10CtxCreateOnDevice**

This function is deprecated as of CUDA 5.0.


**Global cuD3D10GetDirect3DDevice**

This function is deprecated as of CUDA 5.0.


**Global cuD3D10MapResources**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10RegisterResource**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10ResourceGetMappedArray**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10ResourceGetMappedPitch**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10ResourceGetMappedPointer**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10ResourceGetMappedSize**

This function is deprecated as of CUDA 3.0.


CUDA Driver API TRM-06703-001 _vRelease Version  |  763


Deprecated List


**Global cuD3D10ResourceGetSurfaceDimensions**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10ResourceSetMapFlags**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10UnmapResources**

This function is deprecated as of CUDA 3.0.


**Global cuD3D10UnregisterResource**

This function is deprecated as of CUDA 3.0.


**Global cuD3D11CtxCreate**

This function is deprecated as of CUDA 5.0.


**Global cuD3D11CtxCreateOnDevice**

This function is deprecated as of CUDA 5.0.


**Global cuD3D11GetDirect3DDevice**

This function is deprecated as of CUDA 5.0.


CUDA Driver API TRM-06703-001 _vRelease Version  |  764


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