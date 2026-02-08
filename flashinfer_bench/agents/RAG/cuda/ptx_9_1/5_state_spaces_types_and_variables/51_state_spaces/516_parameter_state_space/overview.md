# 5.1.6. Parameter State Space

### 5.1.6. [Parameter State Space](https://docs.nvidia.com/cuda/parallel-thread-execution/#parameter-state-space)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#parameter-state-space "Permalink to this headline")

The parameter (`.param`) state space is used (1) to pass input arguments from the host to the
kernel, (2a) to declare formal input and return parameters for device functions called from within
kernel execution, and (2b) to declare locally-scoped byte array variables that serve as function
call arguments, typically for passing large structures by value to a function. Kernel function
parameters differ from device function parameters in terms of access and sharing (read-only versus
read-write, per-kernel versus per-thread). Note that PTX ISA versions 1.x supports only kernel
function parameters in .param space; device function parameters were previously restricted to the
register state space. The use of parameter state space for device function parameters was introduced
in PTX ISA version 2.0 and requires target architecture `sm_20` or higher. Additional sub-qualifiers
`::entry` or `::func` can be specified on instructions with `.param` state space to indicate
whether the address refers to kernel function parameter or device function parameter. If no
sub-qualifier is specified with the `.param` state space, then the default sub-qualifier is specific
to and dependent on the exact instruction. For example, `st.param` is equivalent to `st.param::func`
whereas `isspacep.param` is equivalent to `isspacep.param::entry`. Refer to the instruction
description for more details on default sub-qualifier assumption.

Note

The location of parameter space is implementation specific. For example, in some implementations
kernel parameters reside in global memory. No access protection is provided between parameter and
global space in this case. Though the exact location of the kernel parameter space is
implementation specific, the kernel parameter space window is always contained within the global
space window. Similarly, function parameters are mapped to parameter passing registers and/or
stack locations based on the function calling conventions of the *Application Binary Interface
(ABI)*. Therefore, PTX code should make no assumptions about the relative locations or ordering
of `.param` space variables.