# 10.8.1.10. tex2DGrad() for sparse CUDA arrays

#### 10.8.1.10. tex2DGrad() for sparse CUDA arrays[](#tex2dgrad-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex2DGrad(cudaTextureObject_t texObj, float x, float y,
        float2 dx, float2 dy, bool* isResident);
```

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)`. The level-of-detail is derived from the `dx` and `dy` gradients. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.