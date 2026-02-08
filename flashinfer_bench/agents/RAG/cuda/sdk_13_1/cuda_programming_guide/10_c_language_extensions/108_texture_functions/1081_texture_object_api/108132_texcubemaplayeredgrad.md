# 10.8.1.32. texCubemapLayeredGrad()

#### 10.8.1.32. texCubemapLayeredGrad()[](#texcubemaplayeredgrad "Permalink to this headline")

```
template<class T>
T texCubemapLayeredGrad(cudaTextureObject_t texObj, float x, float y, float z,
                       int layer, float4 dx, float4 dy);
```

fetches from the CUDA array specified by the cubemap layered texture object `texObj` using texture coordinate `(x,y,z)` and index `layer`, as described in [Cubemap Layered Textures](#cubemap-layered-textures), at level-of-detail derived from the `dx` and `dy` gradients.