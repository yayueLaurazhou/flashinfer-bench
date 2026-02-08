# 11.7. Cluster Dimension Directives

## 11.7. [Cluster Dimension Directives](https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-dimension-directives)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-dimension-directives "Permalink to this headline")

The following directives specify information about clusters:

* `.reqnctapercluster`
* `.explicitcluster`
* `.maxclusterrank`

The `.reqnctapercluster` directive specifies the number of CTAs in the cluster. The
`.explicitcluster` directive specifies that the kernel should be launched with explicit cluster
details. The `.maxclusterrank` directive specifies the maximum number of CTAs in the cluster.

The cluster dimension directives can be applied only on kernel functions.