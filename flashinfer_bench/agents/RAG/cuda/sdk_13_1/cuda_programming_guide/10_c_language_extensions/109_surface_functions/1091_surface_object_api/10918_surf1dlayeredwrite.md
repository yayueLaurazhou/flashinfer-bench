# 10.9.1.8. surf1DLayeredwrite()

#### 10.9.1.8. surf1DLayeredwrite()[](#surf1dlayeredwrite "Permalink to this headline")

```
template<class Type>
void surf1DLayeredwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
```

writes value data to the CUDA array specified by the two-dimensional layered surface object `surfObj` at byte coordinate x and index `layer`.