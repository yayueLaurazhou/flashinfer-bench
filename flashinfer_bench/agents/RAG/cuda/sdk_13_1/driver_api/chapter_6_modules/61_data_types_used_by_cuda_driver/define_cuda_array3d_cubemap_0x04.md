# #define CUDA_ARRAY3D_CUBEMAP 0x04

If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube. The width of such
a CUDA array must be equal to its height, and Depth must be six. If CUDA_ARRAY3D_LAYERED
flag is also set, then the CUDA array is a collection of cubemaps and Depth must be a multiple of six.