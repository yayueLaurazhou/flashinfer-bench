# 10.8.1.29. texCubemapGrad()

#### 10.8.1.29. texCubemapGrad()[](#texcubemapgrad "Permalink to this headline")

```
template<class T>
T texCubemapGrad(cudaTextureObject_t texObj, float x, float, y, float z,
                float4 dx, float4 dy);
```

fetches from the CUDA array specified by the cubemap texture object `texObj` using texture coordinate `(x,y,z)` as described in [Cubemap Textures](#cubemap-textures). The level-of-detail used is derived from the `dx` and `dy` gradients.