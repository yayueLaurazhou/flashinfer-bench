# 10.9.1.4. surf2Dwrite()

#### 10.9.1.4. surf2Dwrite()[](#surf2dwrite "Permalink to this headline")

```
template<class T>
void surf2Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y,
                  boundaryMode = cudaBoundaryModeTrap);
```

writes value data to the CUDA array specified by the two-dimensional surface object `surfObj` at byte coordinate x and y.