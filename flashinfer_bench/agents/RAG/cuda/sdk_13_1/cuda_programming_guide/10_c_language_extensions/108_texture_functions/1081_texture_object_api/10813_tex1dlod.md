# 10.8.1.3. tex1DLod()

#### 10.8.1.3. tex1DLod()[](#tex1dlod "Permalink to this headline")

```
template<class T>
T tex1DLod(cudaTextureObject_t texObj, float x, float level);
```

fetches from the CUDA array specified by the one-dimensional texture object `texObj` using texture coordinate `x` at the level-of-detail `level`.