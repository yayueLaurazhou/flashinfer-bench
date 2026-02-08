# 10.8.1.24. tex2DLayeredLod()

#### 10.8.1.24. tex2DLayeredLod()[](#tex2dlayeredlod "Permalink to this headline")

```
template<class T>
T tex2DLayeredLod(cudaTextureObject_t texObj, float x, float y, int layer,
                  float level);
```

fetches from the CUDA array specified by the two-dimensional [layered texture](#layered-textures) at layer `layer` using texture coordinate `(x,y)`.