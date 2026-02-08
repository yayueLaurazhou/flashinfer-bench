# __host__cudaError_t cudaD3D9ResourceGetSurfaceDimensions (size_t *pWidth, size_t *pHeight, size_t *pDepth, IDirect3DResource9 *pResource, unsigned int face, unsigned int level)

Get the dimensions of a registered Direct3D surface.

##### Parameters

**pWidth**

  - Returned width of surface
**pHeight**

  - Returned height of surface
**pDepth**

  - Returned depth of surface
**pResource**

  - Registered resource to access
**face**

  - Face of resource to access
**level**

  - Level of resource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Returns in *pWidth, *pHeight, and *pDepth the dimensions of the subresource of the mapped
Direct3D resource pResource which corresponds to face and level.

Since anti-aliased surfaces may have multiple samples per pixel, it is possible that the dimensions of a
resource will be an integer factor larger than the dimensions reported by the Direct3D runtime.

The parameters pWidth, pHeight, and pDepth are optional. For 2D surfaces, the value returned in
*pDepth will be 0.

If pResource is not of type IDirect3DBaseTexture9 or IDirect3DSurface9 or if pResource has not
been registered for use with CUDA, then cudaErrorInvalidResourceHandle is returned.

For usage requirements of face and level parameters, see cudaD3D9ResourceGetMappedPointer.


CUDA Runtime API vRelease Version  |  256


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsSubResourceGetMappedArray