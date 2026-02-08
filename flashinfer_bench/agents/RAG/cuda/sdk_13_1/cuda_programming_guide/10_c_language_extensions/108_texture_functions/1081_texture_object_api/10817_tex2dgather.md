# 10.8.1.7. tex2Dgather()

#### 10.8.1.7. tex2Dgather()[](#tex2dgather "Permalink to this headline")

```
template<class T>
T tex2Dgather(cudaTextureObject_t texObj,
              float x, float y, int comp = 0);
```

fetches from the CUDA array specified by the 2D texture object `texObj` using texture coordinates `x` and `y` and the `comp` parameter as described in [Texture Gather](#texture-gather).