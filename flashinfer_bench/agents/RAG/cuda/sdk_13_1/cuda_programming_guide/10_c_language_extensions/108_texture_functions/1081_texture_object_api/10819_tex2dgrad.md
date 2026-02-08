# 10.8.1.9. tex2DGrad()

#### 10.8.1.9. tex2DGrad()[](#tex2dgrad "Permalink to this headline")

```
template<class T>
T tex2DGrad(cudaTextureObject_t texObj, float x, float y,
            float2 dx, float2 dy);
```

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)`. The level-of-detail is derived from the `dx` and `dy` gradients.