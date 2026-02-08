# 5.5.4. im2col mode

### 5.5.4. [`im2col` mode](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode "Permalink to this headline")

Im2col mode supports the following tensor dimensions : 3D, 4D and 5D. In this mode, the tensor data
is treated as a batch of images with the following properties:

* N : number of images in the batch
* D, H, W : size of a 3D image (depth, height and width)
* C: channels per image element

The above properties are associated with 3D, 4D and 5D tensors as follows:

| Dimension | N/D/H/W/C applicability |
| --- | --- |
| 3D | NWC |
| 4D | NHWC |
| 5D | NDHWC |