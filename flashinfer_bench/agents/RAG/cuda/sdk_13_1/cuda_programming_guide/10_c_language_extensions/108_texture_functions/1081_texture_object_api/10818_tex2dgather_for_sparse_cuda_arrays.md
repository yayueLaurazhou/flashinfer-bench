# 10.8.1.8. tex2Dgather() for sparse CUDA arrays

#### 10.8.1.8. tex2Dgather() for sparse CUDA arrays[](#tex2dgather-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex2Dgather(cudaTextureObject_t texObj,
            float x, float y, bool* isResident, int comp = 0);
```

fetches from the CUDA array specified by the 2D texture object `texObj` using texture coordinates `x` and `y` and the `comp` parameter as described in [Texture Gather](#texture-gather). Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.