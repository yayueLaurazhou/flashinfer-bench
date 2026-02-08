# 8.1. Scope and applicability of the model

## 8.1. [Scope and applicability of the model](https://docs.nvidia.com/cuda/parallel-thread-execution/#scope-and-applicability)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#scope-and-applicability "Permalink to this headline")

The constraints specified under this model apply to PTX programs with any PTX ISA version number,
running on `sm_70` or later architectures.

The memory consistency model does not apply to texture (including `ld.global.nc`) and surface
accesses.