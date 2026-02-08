# enum cudaClusterSchedulingPolicy

Cluster scheduling policies. These may be passed to cudaFuncSetAttribute

##### Values

**cudaClusterSchedulingPolicyDefault = 0**
the default policy
**cudaClusterSchedulingPolicySpread = 1**
spread the blocks within a cluster to the SMs
**cudaClusterSchedulingPolicyLoadBalancing = 2**
allow the hardware to load-balance the blocks in a cluster to the SMs