# __host__cudaCreateChannelDesc (int x, int y, int z, int w, cudaChannelFormatKind f)

Returns a channel descriptor using the specified format.

##### Parameters

**x**

  - X component
**y**

  - Y component
**z**

  - Z component
**w**

  - W component
**f**

  - Channel format

##### Returns

Channel descriptor with format f

##### Description

Returns a channel descriptor with format f and number of bits of each component x, y, z, and w. The
cudaChannelFormatDesc is defined as:


where cudaChannelFormatKind is one of cudaChannelFormatKindSigned,
cudaChannelFormatKindUnsigned, or cudaChannelFormatKindFloat.


See also:

cudaCreateChannelDesc ( C++ API), cudaGetChannelDesc, cudaCreateTextureObject,
cudaCreateSurfaceObject


CUDA Runtime API vRelease Version  |  303


Modules