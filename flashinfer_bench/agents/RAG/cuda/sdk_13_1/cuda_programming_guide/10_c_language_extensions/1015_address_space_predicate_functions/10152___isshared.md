# 10.15.2. __isShared()

### 10.15.2. \_\_isShared()[](#isshared "Permalink to this headline")

```
__device__ unsigned int __isShared(const void *ptr);
```

Returns 1 if `ptr` contains the generic address of an object in shared memory space, otherwise returns 0.