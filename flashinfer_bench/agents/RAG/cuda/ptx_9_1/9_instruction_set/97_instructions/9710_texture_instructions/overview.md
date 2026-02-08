# 9.7.10. Texture Instructions

### 9.7.10. [Texture Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#texture-instructions)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#texture-instructions "Permalink to this headline")

This section describes PTX instructions for accessing textures and samplers. PTX supports the
following operations on texture and sampler descriptors:

* Static initialization of texture and sampler descriptors.
* Module-scope and per-entry scope definitions of texture and sampler descriptors.
* Ability to query fields within texture and sampler descriptors.