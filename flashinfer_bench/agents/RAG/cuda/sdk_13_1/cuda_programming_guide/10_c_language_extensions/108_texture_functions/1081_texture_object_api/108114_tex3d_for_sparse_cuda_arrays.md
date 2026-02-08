# 10.8.1.14. tex3D() for sparse CUDA arrays

#### 10.8.1.14. tex3D() for sparse CUDA arrays[](#tex3d-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex3D(cudaTextureObject_t texObj, float x, float y, float z, bool* isResident);
```

fetches from the CUDA array specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.