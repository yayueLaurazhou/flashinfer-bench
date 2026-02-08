# 10.18.3. __assume()

### 10.18.3. \_\_assume()[](#assume "Permalink to this headline")

```
void __assume(bool exp)
```

Allows the compiler to assume that the Boolean argument is true. If the argument is not true at run time, then the behavior is undefined. Note that if the argument has side effects, the behavior is unspecified.

Example:

```
 __device__ int get(int *ptr, int idx) {
   __assume(idx <= 2);
   return ptr[idx];
}
```