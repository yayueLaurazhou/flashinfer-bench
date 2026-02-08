# 5.1.6.2. Kernel Function Parameter Attributes

#### 5.1.6.2. [Kernel Function Parameter Attributes](https://docs.nvidia.com/cuda/parallel-thread-execution/#kernel-function-parameter-attributes)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#kernel-function-parameter-attributes "Permalink to this headline")

Kernel function parameters may be declared with an optional .ptr attribute to indicate that a
parameter is a pointer to memory, and also indicate the state space and alignment of the memory
being pointed to. [Kernel Parameter Attribute: .ptr](https://docs.nvidia.com/cuda/parallel-thread-execution/#kernel-parameter-attribute-ptr)
describes the `.ptr` kernel parameter attribute.