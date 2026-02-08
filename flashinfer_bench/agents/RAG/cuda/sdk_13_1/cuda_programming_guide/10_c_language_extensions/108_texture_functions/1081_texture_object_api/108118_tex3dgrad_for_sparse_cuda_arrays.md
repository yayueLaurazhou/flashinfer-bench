# 10.8.1.18. tex3DGrad() for sparse CUDA arrays

#### 10.8.1.18. tex3DGrad() for sparse CUDA arrays[](#tex3dgrad-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex3DGrad(cudaTextureObject_t texObj, float x, float y, float z,
        float4 dx, float4 dy, bool* isResident);
```

fetches from the CUDA array specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)` at a level-of-detail derived from the X and Y gradients `dx` and `dy`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.