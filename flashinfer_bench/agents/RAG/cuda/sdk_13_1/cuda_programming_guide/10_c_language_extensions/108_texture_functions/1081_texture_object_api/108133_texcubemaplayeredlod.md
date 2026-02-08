# 10.8.1.33. texCubemapLayeredLod()

#### 10.8.1.33. texCubemapLayeredLod()[](#texcubemaplayeredlod "Permalink to this headline")

```
template<class T>
T texCubemapLayeredLod(cudaTextureObject_t texObj, float x, float y, float z,
                       int layer, float level);
```

fetches from the CUDA array specified by the cubemap layered texture object `texObj` using texture coordinate `(x,y,z)` and index `layer`, as described in [Cubemap Layered Textures](#cubemap-layered-textures), at level-of-detail level `level`.