# 10.8.1.20. tex1DLayeredLod()

#### 10.8.1.20. tex1DLayeredLod()[](#tex1dlayeredlod "Permalink to this headline")

```
template<class T>
T tex1DLayeredLod(cudaTextureObject_t texObj, float x, int layer, float level);
```

fetches from the CUDA array specified by the one-dimensional [Layered Textures](#layered-textures) at layer `layer` using texture coordinate `x` and level-of-detail `level`.