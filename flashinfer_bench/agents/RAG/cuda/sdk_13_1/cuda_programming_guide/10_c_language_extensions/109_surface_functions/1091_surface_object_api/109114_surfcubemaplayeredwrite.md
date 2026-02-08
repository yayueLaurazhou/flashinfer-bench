# 10.9.1.14. surfCubemapLayeredwrite()

#### 10.9.1.14. surfCubemapLayeredwrite()[](#surfcubemaplayeredwrite "Permalink to this headline")

```
template<class T>
void surfCubemapLayeredwrite(T data,
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
```

writes value data to the CUDA array specified by the cubemap layered object `surfObj` at byte coordinate `x` and `y`, and index `layerFace`.