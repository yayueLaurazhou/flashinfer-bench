# 8.2.3. Multimem Addresses

### 8.2.3. [Multimem Addresses](https://docs.nvidia.com/cuda/parallel-thread-execution/#multimem-addresses)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#multimem-addresses "Permalink to this headline")

A multimem address is a virtual address which points to multiple distinct memory locations across
devices.

Only *multimem.*\* operations are valid on multimem addresses. That is, the behavior of accessing
a multimem address in any other memory operation is undefined.