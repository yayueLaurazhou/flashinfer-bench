# 10.8.1.27. tex2DLayeredGrad() for sparse CUDA arrays

#### 10.8.1.27. tex2DLayeredGrad() for sparse CUDA arrays[](#tex2dlayeredgrad-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex2DLayeredGrad(cudaTextureObject_t texObj, float x, float y, int layer,
                float2 dx, float2 dy, bool* isResident);
```

fetches from the CUDA array specified by the two-dimensional [layered texture](#layered-textures) at layer `layer` using texture coordinate `(x,y)` and a level-of-detail derived from the `dx` and `dy` gradients. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.