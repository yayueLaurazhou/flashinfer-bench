# 6.4.4. Labels and Function Names as Operands

### 6.4.4. [Labels and Function Names as Operands](https://docs.nvidia.com/cuda/parallel-thread-execution/#labels-and-function-names-as-operands)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#labels-and-function-names-as-operands "Permalink to this headline")

Labels and function names can be used only in `bra`/`brx.idx` and `call` instructions
respectively. Function names can be used in `mov` instruction to get the address of the function
into a register, for use in an indirect call.

Beginning in PTX ISA version 3.1, the `mov` instruction may be used to take the address of kernel
functions, to be passed to a system call that initiates a kernel launch from the GPU. This feature
is part of the support for CUDA Dynamic Parallelism. See the *CUDA Dynamic Parallelism Programming
Guide* for details.