# 10.8.1.19. tex1DLayered()

#### 10.8.1.19. tex1DLayered()[](#tex1dlayered "Permalink to this headline")

```
template<class T>
T tex1DLayered(cudaTextureObject_t texObj, float x, int layer);
```

fetches from the CUDA array specified by the one-dimensional texture object `texObj` using texture coordinate `x` and index `layer`, as described in [Layered Textures](#layered-textures).