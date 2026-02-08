# 5.5.3.2. Traversal-Stride

#### 5.5.3.2. [Traversal-Stride](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode-traversal-stride)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode-traversal-stride "Permalink to this headline")

While the Bounding Box is iterating the tensor across a dimension, the traversal stride specifies
the exact number of elements to be skipped. If no jump over is required, default value of 1 must be
specified.

The traversal stride in dimension 0 can be used for the [Interleave layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-interleaved-layout).
For non-interleaved layout, the traversal stride in
dimension 0 must always be 1.

[Figure 8](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode-bb-example) illustrates tensor, tensor size, tensor stride,
Bounding Box size and traversal stride.

![_images/tensor-tiled-mode-bounding-box-example.png](./ptx_files/tensor-tiled-mode-bounding-box-example.png)


Figure 8 Tiled mode bounding box, tensor size and traversal stride[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode-bb-example "Permalink to this image")