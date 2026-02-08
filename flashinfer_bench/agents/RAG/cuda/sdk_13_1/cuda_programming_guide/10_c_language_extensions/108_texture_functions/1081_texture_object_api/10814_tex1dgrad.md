# 10.8.1.4. tex1DGrad()

#### 10.8.1.4. tex1DGrad()[](#tex1dgrad "Permalink to this headline")

```
template<class T>
T tex1DGrad(cudaTextureObject_t texObj, float x, float dx, float dy);
```

fetches from the CUDA array specified by the one-dimensional texture object `texObj` using texture coordinate `x`. The level-of-detail is derived from the X-gradient `dx` and Y-gradient `dy`.