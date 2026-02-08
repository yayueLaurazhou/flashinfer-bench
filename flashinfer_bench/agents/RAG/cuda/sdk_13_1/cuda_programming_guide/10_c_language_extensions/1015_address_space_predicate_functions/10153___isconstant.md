# 10.15.3. __isConstant()

### 10.15.3. \_\_isConstant()[](#isconstant "Permalink to this headline")

```
__device__ unsigned int __isConstant(const void *ptr);
```

Returns 1 if `ptr` contains the generic address of an object in constant memory space, otherwise returns 0.