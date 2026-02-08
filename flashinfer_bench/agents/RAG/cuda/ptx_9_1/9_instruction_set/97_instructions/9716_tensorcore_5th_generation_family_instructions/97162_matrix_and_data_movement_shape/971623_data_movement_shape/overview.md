# 9.7.16.2.3. Data Movement Shape

##### 9.7.16.2.3. [Data Movement Shape](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-movement-shape)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-movement-shape "Permalink to this headline")

The data movement shape indicates the dimension of the data to be moved to or from the
[Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory). These shapes are described as a tuple `lane x size` where:

* `lane` indicates the number of rows in the [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory); and
* `size` indicates the amount of data, in units of bits (b), across the columns in the
  [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory).

The following shapes are supported by various tcgen05 operations:

| Shape | tcgen05.<op> |
| --- | --- |
| `.16x64b`, `.16x128b`, `.16x256b`, `.16x32bx2`, `.32x32b` | `.ld` / `.st` |
| `.4x256b`, `.32x128b`, `.64x128b`, `.128x256b`, `.128x128b` | `.cp` |
| `.31x256b` (implicit) | `.shift` |