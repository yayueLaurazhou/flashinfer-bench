# 10.15.4. __isGridConstant()

### 10.15.4. \_\_isGridConstant()[](#isgridconstant "Permalink to this headline")

```
__device__ unsigned int __isGridConstant(const void *ptr);
```

Returns 1 if `ptr` contains the generic address of a kernel parameter annotated with `__grid_constant__`, otherwise returns 0. Only supported for compute architectures greater than or equal to 7.x or later.