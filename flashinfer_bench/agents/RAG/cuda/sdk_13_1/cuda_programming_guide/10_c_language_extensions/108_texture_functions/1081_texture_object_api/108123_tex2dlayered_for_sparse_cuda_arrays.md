# 10.8.1.23. tex2DLayered() for Sparse CUDA Arrays

#### 10.8.1.23. tex2DLayered() for Sparse CUDA Arrays[](#tex2dlayered-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex2DLayered(cudaTextureObject_t texObj,
            float x, float y, int layer, bool* isResident);
```

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)` and index `layer`, as described in [Layered Textures](#layered-textures). Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.