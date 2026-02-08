# 7. Abstracting the ABI

# 7. [Abstracting the ABI](https://docs.nvidia.com/cuda/parallel-thread-execution/#abstracting-abi)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#abstracting-abi "Permalink to this headline")

Rather than expose details of a particular calling convention, stack layout, and Application Binary
Interface (ABI), PTX provides a slightly higher-level abstraction and supports multiple ABI
implementations. In this section, we describe the features of PTX needed to achieve this hiding of
the ABI. These include syntax for function definitions, function calls, parameter passing, and
memory allocated on the stack (`alloca`).

Refer to *PTX Writers Guide to Interoperability* for details on generating PTX compliant with
Application Binary Interface (ABI) for the CUDA® architecture.