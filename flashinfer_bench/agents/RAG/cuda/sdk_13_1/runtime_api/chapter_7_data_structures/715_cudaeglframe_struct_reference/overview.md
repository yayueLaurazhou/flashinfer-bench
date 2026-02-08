# 7.15. cudaEglFrame Struct Reference

CUDA EGLFrame Descriptor - structure defining one frame of EGL.

Each frame may contain one or more planes depending on whether the surface is Multiplanar or not.
Each plane of EGLFrame is represented by cudaEglPlaneDesc which is defined as:
‎ typedef struct cudaEglPlaneDesc_st {
unsigned int width;
unsigned int height;
unsigned int depth;
unsigned int pitch;


CUDA Runtime API vRelease Version  |  607


Data Structures