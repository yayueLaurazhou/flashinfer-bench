# 10.8.1.25. tex2DLayeredLod() for sparse CUDA arrays

#### 10.8.1.25. tex2DLayeredLod() for sparse CUDA arrays[](#tex2dlayeredlod-for-sparse-cuda-arrays "Permalink to this headline")

```
                template<class T>
T tex2DLayeredLod(cudaTextureObject_t texObj, float x, float y, int layer,
                float level, bool* isResident);
```

fetches from the CUDA array specified by the two-dimensional [layered texture](#layered-textures) at layer `layer` using texture coordinate `(x,y)`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.