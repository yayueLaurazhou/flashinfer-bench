# 10.8.1.22. tex2DLayered()

#### 10.8.1.22. tex2DLayered()[](#tex2dlayered "Permalink to this headline")

```
template<class T>
T tex2DLayered(cudaTextureObject_t texObj,
               float x, float y, int layer);
```

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)` and index `layer`, as described in [Layered Textures](#layered-textures).