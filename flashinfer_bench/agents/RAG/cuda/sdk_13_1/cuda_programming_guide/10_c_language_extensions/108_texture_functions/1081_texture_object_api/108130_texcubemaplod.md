# 10.8.1.30. texCubemapLod()

#### 10.8.1.30. texCubemapLod()[](#texcubemaplod "Permalink to this headline")

```
template<class T>
T texCubemapLod(cudaTextureObject_t texObj, float x, float, y, float z,
                float level);
```

fetches from the CUDA array specified by the cubemap texture object `texObj` using texture coordinate `(x,y,z)` as described in [Cubemap Textures](#cubemap-textures). The level-of-detail used is given by `level`.