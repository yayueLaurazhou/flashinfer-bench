# 6.2.14.1.4. Cubemap Textures

##### 6.2.14.1.4. Cubemap Textures[](#cubemap-textures "Permalink to this headline")

A *cubemap* texture is a special type of two-dimensional layered texture that has six layers representing the faces of a cube:

* The width of a layer is equal to its height.
* The cubemap is addressed using three texture coordinates *x*, *y*, and *z* that are interpreted as a direction vector emanating from the center of the cube and pointing to one face of the cube and a texel within the layer corresponding to that face. More specifically, the face is selected by the coordinate with largest magnitude *m* and the corresponding layer is addressed using coordinates *(s/m+1)/2* and *(t/m+1)/2* where *s* and *t* are defined in [Table 6](#cubemap-textures-cubemap-fetch).

Table 6 Cubemap Fetch[](#cubemap-textures-cubemap-fetch "Permalink to this table")








|  | | face | m | s | t |
| --- | --- | --- | --- | --- | --- |
| `|x| > |y|` and `|x| > |z|` | x ≥ 0 | 0 | x | -z | -y |
| x < 0 | 1 | -x | z | -y |
| `|y| > |x|` and `|y| > |z|` | y ≥ 0 | 2 | y | x | z |
| y < 0 | 3 | -y | x | -z |
| `|z| > |x|` and `|z| > |y|` | z ≥ 0 | 4 | z | x | -y |
| z < 0 | 5 | -z | -x | -y |

A cubemap texture can only be a CUDA array by calling `cudaMalloc3DArray()` with the `cudaArrayCubemap` flag.

Cubemap textures are fetched using the device function described in [texCubemap()](#texcubemap-object).

Cubemap textures are only supported on devices of compute capability 2.0 and higher.