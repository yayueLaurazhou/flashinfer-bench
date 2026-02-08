# 7.55. cudaMemLocation Struct Reference

Specifies a memory location.

To specify a gpu, set type = cudaMemLocationTypeDevice and set id = the gpu's device ordinal. To
specify a cpu NUMA node, set type = cudaMemLocationTypeHostNuma and set id = host NUMA
node id.