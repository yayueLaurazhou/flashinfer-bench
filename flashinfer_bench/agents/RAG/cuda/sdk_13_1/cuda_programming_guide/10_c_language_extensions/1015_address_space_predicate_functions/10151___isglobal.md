# 10.15.1. __isGlobal()

### 10.15.1. \_\_isGlobal()[](#isglobal "Permalink to this headline")

```
__device__ unsigned int __isGlobal(const void *ptr);
```

Returns 1 if `ptr` contains the generic address of an object in global memory space, otherwise returns 0.