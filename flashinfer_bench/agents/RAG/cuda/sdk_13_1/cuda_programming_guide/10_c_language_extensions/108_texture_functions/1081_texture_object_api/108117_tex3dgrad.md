# 10.8.1.17. tex3DGrad()

#### 10.8.1.17. tex3DGrad()[](#tex3dgrad "Permalink to this headline")

```
template<class T>
T tex3DGrad(cudaTextureObject_t texObj, float x, float y, float z,
            float4 dx, float4 dy);
```

fetches from the CUDA array specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)` at a level-of-detail derived from the X and Y gradients `dx` and `dy`.