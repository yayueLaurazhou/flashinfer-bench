# 10.8.1.21. tex1DLayeredGrad()

#### 10.8.1.21. tex1DLayeredGrad()[](#tex1dlayeredgrad "Permalink to this headline")

```
template<class T>
T tex1DLayeredGrad(cudaTextureObject_t texObj, float x, int layer,
                   float dx, float dy);
```

fetches from the CUDA array specified by the one-dimensional [layered texture](#layered-textures) at layer `layer` using texture coordinate `x` and a level-of-detail derived from the `dx` and `dy` gradients.