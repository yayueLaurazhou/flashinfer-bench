# enum CUjit_option

Online compiler and linker options

###### Values

**CU_JIT_MAX_REGISTERS = 0**
Max number of registers that a thread may use. Option type: unsigned int Applies to: compiler only
**CU_JIT_THREADS_PER_BLOCK = 1**
IN: Specifies minimum number of threads per block to target compilation for OUT: Returns the
number of threads the compiler actually targeted. This restricts the resource utilization of the
compiler (e.g. max registers) such that a block with the given number of threads should be able
to launch based on register limitations. Note, this option does not currently take into account
any other resource limitations, such as shared memory utilization. Cannot be combined with
CU_JIT_TARGET. Option type: unsigned int Applies to: compiler only
**CU_JIT_WALL_TIME = 2**
Overwrites the option value with the total wall clock time, in milliseconds, spent in the compiler and
linker Option type: float Applies to: compiler and linker
**CU_JIT_INFO_LOG_BUFFER = 3**
Pointer to a buffer in which to print any log messages that are informational in nature (the buffer
size is specified via option CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES) Option type: char *
Applies to: compiler and linker
**CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4**
IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
OUT: Amount of log buffer filled with messages Option type: unsigned int Applies to: compiler and
linker
**CU_JIT_ERROR_LOG_BUFFER = 5**
Pointer to a buffer in which to print any log messages that reflect errors (the buffer size is specified
via option CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES) Option type: char * Applies to:
compiler and linker
**CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6**
IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
OUT: Amount of log buffer filled with messages Option type: unsigned int Applies to: compiler and
linker
**CU_JIT_OPTIMIZATION_LEVEL = 7**
Level of optimizations to apply to generated code (0 - 4), with 4 being the default and highest level
of optimizations. Option type: unsigned int Applies to: compiler only
**CU_JIT_TARGET_FROM_CUCONTEXT = 8**
No option value required. Determines the target based on the current attached context (default)
Option type: No option value needed Applies to: compiler and linker
**CU_JIT_TARGET = 9**


CUDA Driver API TRM-06703-001 _vRelease Version  |  52


Modules



Target is chosen based on supplied CUjit_target. Cannot be combined with
CU_JIT_THREADS_PER_BLOCK. Option type: unsigned int for enumerated type CUjit_target
Applies to: compiler and linker
**CU_JIT_FALLBACK_STRATEGY = 10**
Specifies choice of fallback strategy if matching cubin is not found. Choice is based on supplied
CUjit_fallback. This option cannot be used with cuLink* APIs as the linker requires exact matches.
Option type: unsigned int for enumerated type CUjit_fallback Applies to: compiler only
**CU_JIT_GENERATE_DEBUG_INFO = 11**
Specifies whether to create debug information in output (-g) (0: false, default) Option type: int
Applies to: compiler and linker
**CU_JIT_LOG_VERBOSE = 12**
Generate verbose log messages (0: false, default) Option type: int Applies to: compiler and linker
**CU_JIT_GENERATE_LINE_INFO = 13**
Generate line number information (-lineinfo) (0: false, default) Option type: int Applies to: compiler
only
**CU_JIT_CACHE_MODE = 14**
Specifies whether to enable caching explicitly (-dlcm) Choice is based on supplied
CUjit_cacheMode_enum. Option type: unsigned int for enumerated type CUjit_cacheMode_enum
Applies to: compiler only
**CU_JIT_NEW_SM3X_OPT = 15**



Deprecated This jit option is deprecated and should not be used.
**CU_JIT_FAST_COMPILE = 16**
This jit option is used for internal purpose only.
**CU_JIT_GLOBAL_SYMBOL_NAMES = 17**
Array of device symbol names that will be relocated to the corresponding host
addresses stored in CU_JIT_GLOBAL_SYMBOL_ADDRESSES. Must contain
CU_JIT_GLOBAL_SYMBOL_COUNT entries. When loading a device module, driver will
relocate all encountered unresolved symbols to the host addresses. It is only allowed to register
symbols that correspond to unresolved global variables. It is illegal to register the same device
symbol at multiple addresses. Option type: const char ** Applies to: dynamic linker only
**CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18**
Array of host addresses that will be used to relocate corresponding device symbols stored in
CU_JIT_GLOBAL_SYMBOL_NAMES. Must contain CU_JIT_GLOBAL_SYMBOL_COUNT
entries. Option type: void ** Applies to: dynamic linker only
**CU_JIT_GLOBAL_SYMBOL_COUNT = 19**
Number of entries in CU_JIT_GLOBAL_SYMBOL_NAMES and
CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays. Option type: unsigned int Applies to:
dynamic linker only
**CU_JIT_LTO = 20**



Deprecated Enable link-time optimization (-dlto) for device code (Disabled by default). This option
is not supported on 32-bit platforms. Option type: int Applies to: compiler and linker Only valid
with LTO-IR compiled with toolkits prior to CUDA 12.0
**CU_JIT_FTZ = 21**



CUDA Driver API TRM-06703-001 _vRelease Version  |  53


Modules



Deprecated Control single-precision denormals (-ftz) support (0: false, default). 1 : flushes denormal
values to zero 0 : preserves denormal values Option type: int Applies to: link-time optimization
specified with CU_JIT_LTO Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
**CU_JIT_PREC_DIV = 22**



Deprecated Control single-precision floating-point division and reciprocals (-prec-div) support (1:
true, default). 1 : Enables the IEEE round-to-nearest mode 0 : Enables the fast approximation mode
Option type: int Applies to: link-time optimization specified with CU_JIT_LTO Only valid with
LTO-IR compiled with toolkits prior to CUDA 12.0
**CU_JIT_PREC_SQRT = 23**



Deprecated Control single-precision floating-point square root (-prec-sqrt) support (1: true, default).
1 : Enables the IEEE round-to-nearest mode 0 : Enables the fast approximation mode Option
type: int Applies to: link-time optimization specified with CU_JIT_LTO Only valid with LTO-IR
compiled with toolkits prior to CUDA 12.0
**CU_JIT_FMA = 24**



Deprecated Enable/Disable the contraction of floating-point multiplies and adds/subtracts into
floating-point multiply-add (-fma) operations (1: Enable, default; 0: Disable). Option type: int
Applies to: link-time optimization specified with CU_JIT_LTO Only valid with LTO-IR compiled
with toolkits prior to CUDA 12.0
**CU_JIT_REFERENCED_KERNEL_NAMES = 25**



Deprecated Array of kernel names that should be preserved at link time while others can be
removed. Must contain CU_JIT_REFERENCED_KERNEL_COUNT entries. Note that kernel
names can be mangled by the compiler in which case the mangled name needs to be specified.
Wildcard "*" can be used to represent zero or more characters instead of specifying the full or
mangled name. It is important to note that the wildcard "*" is also added implicitly. For example,
specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and thus preserve all kernels with those
names. This can be avoided by providing a more specific name like "barfoobaz". Option type: const
char ** Applies to: dynamic linker only Only valid with LTO-IR compiled with toolkits prior to
CUDA 12.0
**CU_JIT_REFERENCED_KERNEL_COUNT = 26**



Deprecated Number of entries in CU_JIT_REFERENCED_KERNEL_NAMES array. Option type:
unsigned int Applies to: dynamic linker only Only valid with LTO-IR compiled with toolkits prior
to CUDA 12.0
**CU_JIT_REFERENCED_VARIABLE_NAMES = 27**



Deprecated Array of variable names (__device__ and/or __constant__) that
should be preserved at link time while others can be removed. Must contain
CU_JIT_REFERENCED_VARIABLE_COUNT entries. Note that variable names can be mangled
by the compiler in which case the mangled name needs to be specified. Wildcard "*" can be used
to represent zero or more characters instead of specifying the full or mangled name. It is important
to note that the wildcard "*" is also added implicitly. For example, specifying "foo" will match
"foobaz", "barfoo", "barfoobaz" and thus preserve all variables with those names. This can be
avoided by providing a more specific name like "barfoobaz". Option type: const char ** Applies to:
link-time optimization specified with CU_JIT_LTO Only valid with LTO-IR compiled with toolkits
prior to CUDA 12.0



CUDA Driver API TRM-06703-001 _vRelease Version  |  54


Modules


**CU_JIT_REFERENCED_VARIABLE_COUNT = 28**

Deprecated Number of entries in CU_JIT_REFERENCED_VARIABLE_NAMES array. Option
type: unsigned int Applies to: link-time optimization specified with CU_JIT_LTO Only valid with
LTO-IR compiled with toolkits prior to CUDA 12.0
**CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29**

Deprecated This option serves as a hint to enable the JIT compiler/linker to remove constant
(__constant__) and device (__device__) variables unreferenced in device code (Disabled
by default). Note that host references to constant and device variables using APIs like
cuModuleGetGlobal() with this option specified may result in undefined behavior unless the
variables are explicitly specified using CU_JIT_REFERENCED_VARIABLE_NAMES. Option
type: int Applies to: link-time optimization specified with CU_JIT_LTO Only valid with LTO-IR
compiled with toolkits prior to CUDA 12.0
**CU_JIT_POSITION_INDEPENDENT_CODE = 30**
Generate position independent code (0: false) Option type: int Applies to: compiler only
**CU_JIT_MIN_CTA_PER_SM = 31**
This option hints to the JIT compiler the minimum number of CTAs from the kernel’s grid to be
mapped to a SM. This option is ignored when used together with CU_JIT_MAX_REGISTERS
or CU_JIT_THREADS_PER_BLOCK. Optimizations based on this option need
CU_JIT_MAX_THREADS_PER_BLOCK to be specified as well. For kernels already
using PTX directive .minnctapersm, this option will be ignored by default. Use
CU_JIT_OVERRIDE_DIRECTIVE_VALUES to let this option take precedence over the PTX
directive. Option type: unsigned int Applies to: compiler only
**CU_JIT_MAX_THREADS_PER_BLOCK = 32**
Maximum number threads in a thread block, computed as the product of the maximum extent
specifed for each dimension of the block. This limit is guaranteed not to be exeeded in any
invocation of the kernel. Exceeding the the maximum number of threads results in runtime error or
kernel launch failure. For kernels already using PTX directive .maxntid, this option will be ignored
by default. Use CU_JIT_OVERRIDE_DIRECTIVE_VALUES to let this option take precedence
over the PTX directive. Option type: int Applies to: compiler only
**CU_JIT_OVERRIDE_DIRECTIVE_VALUES = 33**
This option lets the values specified using CU_JIT_MAX_REGISTERS,
CU_JIT_THREADS_PER_BLOCK, CU_JIT_MAX_THREADS_PER_BLOCK and
CU_JIT_MIN_CTA_PER_SM take precedence over any PTX directives. (0: Disable, default; 1:
Enable) Option type: int Applies to: compiler only
**CU_JIT_SPLIT_COMPILE = 34**
This option specifies the maximum number of concurrent threads to use when running compiler
optimizations. If the specified value is 1, the option will be ignored. If the specified value is 0, the
number of threads will match the number of CPUs on the underlying machine. Otherwise, if the
option is N, then up to N threads will be used. Option type: unsigned int Applies to: compiler only
**CU_JIT_BINARY_LOADER_THREAD_COUNT = 35**
This option specifies the maximum number of concurrent threads to use when compiling
device code. If the specified value is 1, the option will be ignored. If the specified value is 0,
the number of threads will match the number of CPUs on the underlying machine. Otherwise,


CUDA Driver API TRM-06703-001 _vRelease Version  |  55


Modules


if the option is N, then up to N threads will be used. This option is ignored if the env var
CUDA_BINARY_LOADER_THREAD_COUNT is set. Option type: unsigned int Applies to:
compiler and linker
**CU_JIT_NUM_OPTIONS**