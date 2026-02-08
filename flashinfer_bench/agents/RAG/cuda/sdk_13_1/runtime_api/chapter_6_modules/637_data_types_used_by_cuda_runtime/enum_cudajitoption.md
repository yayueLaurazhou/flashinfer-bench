# enum cudaJitOption

Online compiler and linker options

##### Values

**cudaJitMaxRegisters = 0**
Max number of registers that a thread may use. Option type: unsigned int Applies to: compiler only
**cudaJitThreadsPerBlock = 1**
IN: Specifies minimum number of threads per block to target compilation for OUT: Returns the
number of threads the compiler actually targeted. This restricts the resource utilization of the
compiler (e.g. max registers) such that a block with the given number of threads should be able
to launch based on register limitations. Note, this option does not currently take into account any
other resource limitations, such as shared memory utilization. Option type: unsigned int Applies to:
compiler only
**cudaJitWallTime = 2**
Overwrites the option value with the total wall clock time, in milliseconds, spent in the compiler and
linker Option type: float Applies to: compiler and linker
**cudaJitInfoLogBuffer = 3**
Pointer to a buffer in which to print any log messages that are informational in nature (the buffer
size is specified via option cudaJitInfoLogBufferSizeBytes) Option type: char * Applies to:
compiler and linker
**cudaJitInfoLogBufferSizeBytes = 4**
IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
OUT: Amount of log buffer filled with messages Option type: unsigned int Applies to: compiler and
linker
**cudaJitErrorLogBuffer = 5**
Pointer to a buffer in which to print any log messages that reflect errors (the buffer size is specified
via option cudaJitErrorLogBufferSizeBytes) Option type: char * Applies to: compiler and linker
**cudaJitErrorLogBufferSizeBytes = 6**
IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
OUT: Amount of log buffer filled with messages Option type: unsigned int Applies to: compiler and
linker


CUDA Runtime API vRelease Version  |  564


Modules


**cudaJitOptimizationLevel = 7**
Level of optimizations to apply to generated code (0 - 4), with 4 being the default and highest level
of optimizations. Option type: unsigned int Applies to: compiler only
**cudaJitFallbackStrategy = 10**
Specifies choice of fallback strategy if matching cubin is not found. Choice is based on supplied
cudaJit_Fallback. Option type: unsigned int for enumerated type cudaJit_Fallback Applies to:
compiler only
**cudaJitGenerateDebugInfo = 11**
Specifies whether to create debug information in output (-g) (0: false, default) Option type: int
Applies to: compiler and linker
**cudaJitLogVerbose = 12**
Generate verbose log messages (0: false, default) Option type: int Applies to: compiler and linker
**cudaJitGenerateLineInfo = 13**
Generate line number information (-lineinfo) (0: false, default) Option type: int Applies to: compiler
only
**cudaJitCacheMode = 14**
Specifies whether to enable caching explicitly (-dlcm) Choice is based on supplied
cudaJit_CacheMode. Option type: unsigned int for enumerated type cudaJit_CacheMode Applies to:
compiler only
**cudaJitPositionIndependentCode = 30**
Generate position independent code (0: false) Option type: int Applies to: compiler only
**cudaJitMinCtaPerSm = 31**
This option hints to the JIT compiler the minimum number of CTAs from the kernel’s grid to
be mapped to a SM. This option is ignored when used together with cudaJitMaxRegisters or
cudaJitThreadsPerBlock. Optimizations based on this option need cudaJitMaxThreadsPerBlock to
be specified as well. For kernels already using PTX directive .minnctapersm, this option will be
ignored by default. Use cudaJitOverrideDirectiveValues to let this option take precedence over the
PTX directive. Option type: unsigned int Applies to: compiler only
**cudaJitMaxThreadsPerBlock = 32**
Maximum number threads in a thread block, computed as the product of the maximum extent
specifed for each dimension of the block. This limit is guaranteed not to be exeeded in any
invocation of the kernel. Exceeding the the maximum number of threads results in runtime error or
kernel launch failure. For kernels already using PTX directive .maxntid, this option will be ignored
by default. Use cudaJitOverrideDirectiveValues to let this option take precedence over the PTX
directive. Option type: int Applies to: compiler only
**cudaJitOverrideDirectiveValues = 33**
This option lets the values specified using cudaJitMaxRegisters, cudaJitThreadsPerBlock,
cudaJitMaxThreadsPerBlock and cudaJitMinCtaPerSm take precedence over any PTX directives.
(0: Disable, default; 1: Enable) Option type: int Applies to: compiler only