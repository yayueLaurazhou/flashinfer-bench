# enum CUjitInputType

Device code formats

###### Values

**CU_JIT_INPUT_CUBIN = 0**
Compiled device-class-specific device code Applicable options: none
**CU_JIT_INPUT_PTX = 1**
PTX source code Applicable options: PTX compiler options
**CU_JIT_INPUT_FATBINARY = 2**
Bundle of multiple cubins and/or PTX of some device code Applicable options: PTX compiler
options, CU_JIT_FALLBACK_STRATEGY
**CU_JIT_INPUT_OBJECT = 3**
Host object with embedded device code Applicable options: PTX compiler options,
CU_JIT_FALLBACK_STRATEGY
**CU_JIT_INPUT_LIBRARY = 4**
Archive of host objects with embedded device code Applicable options: PTX compiler options,
CU_JIT_FALLBACK_STRATEGY
**CU_JIT_INPUT_NVVM = 5**

Deprecated High-level intermediate code for link-time optimization Applicable options: NVVM
compiler options, PTX compiler options Only valid with LTO-IR compiled with toolkits prior to
CUDA 12.0
**CU_JIT_NUM_INPUT_TYPES = 6**