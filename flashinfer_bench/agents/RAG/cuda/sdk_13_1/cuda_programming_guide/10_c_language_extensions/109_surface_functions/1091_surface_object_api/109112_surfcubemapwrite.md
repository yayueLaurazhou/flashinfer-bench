# 10.9.1.12. surfCubemapwrite()

#### 10.9.1.12. surfCubemapwrite()[](#surfcubemapwrite "Permalink to this headline")

```
template<class T>
void surfCubemapwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);
```

writes value data to the CUDA array specified by the cubemap object `surfObj` at byte coordinate x and y, and face index face.