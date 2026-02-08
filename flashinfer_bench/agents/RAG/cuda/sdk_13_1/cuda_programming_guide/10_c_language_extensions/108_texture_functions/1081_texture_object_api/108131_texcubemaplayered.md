# 10.8.1.31. texCubemapLayered()

#### 10.8.1.31. texCubemapLayered()[](#texcubemaplayered "Permalink to this headline")

```
template<class T>
T texCubemapLayered(cudaTextureObject_t texObj,
                    float x, float y, float z, int layer);
```

fetches from the CUDA array specified by the cubemap layered texture object `texObj` using texture coordinates `(x,y,z)`, and index `layer`, as described in [Cubemap Layered Textures](#cubemap-layered-textures).