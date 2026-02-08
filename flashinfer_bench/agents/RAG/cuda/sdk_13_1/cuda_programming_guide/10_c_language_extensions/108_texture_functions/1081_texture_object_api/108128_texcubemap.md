# 10.8.1.28. texCubemap()

#### 10.8.1.28. texCubemap()[](#texcubemap "Permalink to this headline")

```
template<class T>
T texCubemap(cudaTextureObject_t texObj, float x, float y, float z);
```

fetches the CUDA array specified by the cubemap texture object `texObj` using texture coordinate `(x,y,z)`, as described in [Cubemap Textures](#cubemap-textures).