# 10.8.1.6. tex2D() for sparse CUDA arrays

#### 10.8.1.6. tex2D() for sparse CUDA arrays[](#tex2d-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex2D(cudaTextureObject_t texObj, float x, float y, bool* isResident);
```

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.