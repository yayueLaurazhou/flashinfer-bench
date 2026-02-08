# 10.17.3. Example

### 10.17.3. Example[](#example "Permalink to this headline")

```
__device__ void foo(unsigned int num) {
    int4 *ptr = (int4 *)alloca(num * sizeof(int4));
    // use of ptr
    ...
}
```