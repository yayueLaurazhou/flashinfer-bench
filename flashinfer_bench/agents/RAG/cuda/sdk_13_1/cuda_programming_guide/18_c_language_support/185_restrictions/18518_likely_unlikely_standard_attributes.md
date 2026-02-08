# 18.5.18. [[likely]] / [[unlikely]] Standard Attributes

### 18.5.18. [[likely]] / [[unlikely]] Standard Attributes[](#likely-unlikely-standard-attributes "Permalink to this headline")

These attributes are accepted in all configurations that support the C++ standard attribute syntax. The attributes can be used to hint to the device compiler optimizer whether a statement is more or less likely to be executed compared to any alternative path that does not include the statement.

Example:

```
__device__ int foo(int x) {

 if (i < 10) [[likely]] { // the 'if' block will likely be entered
  return 4;
 }
 if (i < 20) [[unlikely]] { // the 'if' block will not likely be entered
  return 1;
 }
 return 0;
}
```

If these attributes are used in host code when `__CUDA_ARCH__` is undefined, then they will be present in the code parsed by the host compiler, which may generate a warning if the attributes are not supported. For example, `clang`11 host compiler will generate an ‘unknown attribute’ warning.