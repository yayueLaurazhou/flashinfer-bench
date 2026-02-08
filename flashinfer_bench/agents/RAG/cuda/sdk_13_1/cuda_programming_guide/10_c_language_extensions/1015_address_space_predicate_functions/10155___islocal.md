# 10.15.5. __isLocal()

### 10.15.5. \_\_isLocal()[](#islocal "Permalink to this headline")

```
__device__ unsigned int __isLocal(const void *ptr);
```

Returns 1 if `ptr` contains the generic address of an object in local memory space, otherwise returns 0.