# enum cudaLaunchMemSyncDomain

Memory Synchronization Domain

A kernel can be launched in a specified memory synchronization domain that affects all memory
operations issued by that kernel. A memory barrier issued in one domain will only order memory
operations in that domain, thus eliminating latency increase from memory barriers ordering unrelated
traffic.

By default, kernels are launched in domain 0. Kernel launched with
cudaLaunchMemSyncDomainRemote will have a different domain ID. User may also alter the
domain ID with cudaLaunchMemSyncDomainMap for a specific stream / graph node / kernel
launch. See cudaLaunchAttributeMemSyncDomain, cudaStreamSetAttribute, cudaLaunchKernelEx,
cudaGraphKernelNodeSetAttribute.


CUDA Runtime API vRelease Version  |  568


Modules


Memory operations done in kernels launched in different domains are considered system-scope
distanced. In other words, a GPU scoped memory synchronization is not sufficient for memory order to
be observed by kernels in another memory synchronization domain even if they are on the same GPU.

##### Values

**cudaLaunchMemSyncDomainDefault = 0**
Launch kernels in the default domain
**cudaLaunchMemSyncDomainRemote = 1**
Launch kernels in the remote domain