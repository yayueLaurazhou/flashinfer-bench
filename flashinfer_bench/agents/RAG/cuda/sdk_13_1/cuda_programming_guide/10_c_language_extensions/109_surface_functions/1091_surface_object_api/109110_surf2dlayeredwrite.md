# 10.9.1.10. surf2DLayeredwrite()

#### 10.9.1.10. surf2DLayeredwrite()[](#surf2dlayeredwrite "Permalink to this headline")

```
template<class T>
void surf2DLayeredwrite(T data,
                          cudaSurfaceObject_t surfObj,
                          int x, int y, int layer,
                          boundaryMode = cudaBoundaryModeTrap);
```

writes value data to the CUDA array specified by the one-dimensional layered surface object `surfObj` at byte coordinate x and y, and index `layer`.