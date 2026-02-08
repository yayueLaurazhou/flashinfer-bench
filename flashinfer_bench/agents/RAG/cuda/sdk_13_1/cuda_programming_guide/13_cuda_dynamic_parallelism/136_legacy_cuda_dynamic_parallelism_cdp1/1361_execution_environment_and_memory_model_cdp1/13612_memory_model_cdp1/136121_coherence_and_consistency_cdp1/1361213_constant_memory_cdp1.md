# 13.6.1.2.1.3. Constant Memory (CDP1)

###### 13.6.1.2.1.3. Constant Memory (CDP1)[](#constant-memory-cdp1 "Permalink to this headline")

See [Constant Memory](#constant-memory), above, for CDP2 version of document.

Constants are immutable and may not be modified from the device, even between parent and child launches. That is to say, the value of all `__constant__` variables must be set from the host prior to launch. Constant memory is inherited automatically by all child kernels from their respective parents.

Taking the address of a constant memory object from within a kernel thread has the same semantics as for all CUDA programs, and passing that pointer from parent to child or from a child to parent is naturally supported.