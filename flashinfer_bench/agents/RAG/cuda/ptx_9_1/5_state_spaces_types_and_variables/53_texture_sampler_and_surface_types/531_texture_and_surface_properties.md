# 5.3.1. Texture and Surface Properties

### 5.3.1. [Texture and Surface Properties](https://docs.nvidia.com/cuda/parallel-thread-execution/#texture-surface-properties)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#texture-surface-properties "Permalink to this headline")

Fields `width`, `height`, and `depth` specify the size of the texture or surface in number of
elements in each dimension.

The `channel_data_type` and `channel_order` fields specify these properties of the texture or
surface using enumeration types corresponding to the source language API. For example, see
[Channel Data Type and Channel Order Fields](https://docs.nvidia.com/cuda/parallel-thread-execution/#channel-data-type-and-channel-order-fields) for
the OpenCL enumeration types currently supported in PTX.