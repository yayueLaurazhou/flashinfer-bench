# #define CUDA_ARRAY3D_LAYERED 0x01

If set, the CUDA array is a collection of layers, where each layer is either a 1D or a 2D array and the
Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number of layers, not the depth of
a 3D array.