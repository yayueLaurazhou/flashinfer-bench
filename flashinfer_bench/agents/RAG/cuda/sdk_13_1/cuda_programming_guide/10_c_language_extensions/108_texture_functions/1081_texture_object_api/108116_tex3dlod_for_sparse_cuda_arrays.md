# 10.8.1.16. tex3DLod() for sparse CUDA arrays

#### 10.8.1.16. tex3DLod() for sparse CUDA arrays[](#tex3dlod-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex3DLod(cudaTextureObject_t texObj, float x, float y, float z, float level, bool* isResident);
```

fetches from the CUDA array or the region of linear memory specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)` at level-of-detail `level`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.