# 10.8.1.26. tex2DLayeredGrad()

#### 10.8.1.26. tex2DLayeredGrad()[](#tex2dlayeredgrad "Permalink to this headline")

```
template<class T>
T tex2DLayeredGrad(cudaTextureObject_t texObj, float x, float y, int layer,
                   float2 dx, float2 dy);
```

fetches from the CUDA array specified by the two-dimensional [layered texture](#layered-textures) at layer `layer` using texture coordinate `(x,y)` and a level-of-detail derived from the `dx` and `dy` gradients.