# 7.45. cudaLaunchMemSyncDomainMap Struct Reference

Memory Synchronization Domain map

See cudaLaunchMemSyncDomain.

By default, kernels are launched in domain 0. Kernel launched with
cudaLaunchMemSyncDomainRemote will have a different domain ID. User may also alter the domain
ID with cudaLaunchMemSyncDomainMap for a specific stream / graph node / kernel launch. See
cudaLaunchAttributeMemSyncDomainMap.

Domain ID range is available through cudaDevAttrMemSyncDomainCount.


CUDA Runtime API vRelease Version  |  632


Data Structures