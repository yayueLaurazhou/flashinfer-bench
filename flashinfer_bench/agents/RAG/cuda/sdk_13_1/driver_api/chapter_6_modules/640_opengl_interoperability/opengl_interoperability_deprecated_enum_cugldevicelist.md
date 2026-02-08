# OpenGL Interoperability [DEPRECATED] enum CUGLDeviceList

CUDA devices corresponding to an OpenGL device

###### Values

**CU_GL_DEVICE_LIST_ALL = 0x01**
The CUDA devices for all GPUs used by the current OpenGL context
**CU_GL_DEVICE_LIST_CURRENT_FRAME = 0x02**


CUDA Driver API TRM-06703-001 _vRelease Version  |  599


Modules


The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering
frame
**CU_GL_DEVICE_LIST_NEXT_FRAME = 0x03**
The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame