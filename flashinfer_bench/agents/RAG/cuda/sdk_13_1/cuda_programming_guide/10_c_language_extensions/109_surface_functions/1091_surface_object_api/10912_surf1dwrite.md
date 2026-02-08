# 10.9.1.2. surf1Dwrite

#### 10.9.1.2. surf1Dwrite[](#surf1dwrite "Permalink to this headline")

```
template<class T>
void surf1Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x,
                  boundaryMode = cudaBoundaryModeTrap);
```

writes value data to the CUDA array specified by the one-dimensional surface object `surfObj` at byte coordinate x.