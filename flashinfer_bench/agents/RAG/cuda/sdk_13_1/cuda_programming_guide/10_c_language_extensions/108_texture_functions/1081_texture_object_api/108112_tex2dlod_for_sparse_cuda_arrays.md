# 10.8.1.12. tex2DLod() for sparse CUDA arrays

#### 10.8.1.12. tex2DLod() for sparse CUDA arrays[](#tex2dlod-for-sparse-cuda-arrays "Permalink to this headline")

```
        template<class T>
tex2DLod(cudaTextureObject_t texObj, float x, float y, float level, bool* isResident);
```

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)` at level-of-detail `level`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.