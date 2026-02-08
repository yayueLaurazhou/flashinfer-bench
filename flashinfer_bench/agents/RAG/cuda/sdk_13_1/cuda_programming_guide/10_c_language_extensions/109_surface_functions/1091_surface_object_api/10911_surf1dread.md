# 10.9.1.1. surf1Dread()

#### 10.9.1.1. surf1Dread()[](#surf1dread "Permalink to this headline")

```
template<class T>
T surf1Dread(cudaSurfaceObject_t surfObj, int x,
               boundaryMode = cudaBoundaryModeTrap);
```

reads the CUDA array specified by the one-dimensional surface object `surfObj` using byte coordinate x.