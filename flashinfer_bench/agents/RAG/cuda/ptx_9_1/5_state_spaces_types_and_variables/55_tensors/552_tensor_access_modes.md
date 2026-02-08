# 5.5.2. Tensor Access Modes

### 5.5.2. [Tensor Access Modes](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-access-modes)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-access-modes "Permalink to this headline")

Tensor data can be accessed in two modes:

* Tiled mode:

  In tiled mode, the source multi-dimensional tensor layout is preserved at the destination.
* Im2col mode:

  In im2col mode, the elements in the Bounding Box of the source tensor are rearranged into columns
  at the destination. Refer [here](https://in.mathworks.com/help/images/ref/im2col.html) for more details.