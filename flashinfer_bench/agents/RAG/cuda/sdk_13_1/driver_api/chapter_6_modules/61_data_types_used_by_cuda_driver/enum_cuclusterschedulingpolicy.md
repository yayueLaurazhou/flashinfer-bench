# enum CUclusterSchedulingPolicy

Cluster scheduling policies. These may be passed to cuFuncSetAttribute or cuKernelSetAttribute


CUDA Driver API TRM-06703-001 _vRelease Version  |  21


Modules

###### Values

**CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0**
the default policy
**CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1**
spread the blocks within a cluster to the SMs
**CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2**
allow the hardware to load-balance the blocks in a cluster to the SMs