# 10.9.1.6. surf3Dwrite()

#### 10.9.1.6. surf3Dwrite()[](#surf3dwrite "Permalink to this headline")

```
template<class T>
void surf3Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y, int z,
                  boundaryMode = cudaBoundaryModeTrap);
```

writes value data to the CUDA array specified by the three-dimensional object `surfObj` at byte coordinate x, y, and z.