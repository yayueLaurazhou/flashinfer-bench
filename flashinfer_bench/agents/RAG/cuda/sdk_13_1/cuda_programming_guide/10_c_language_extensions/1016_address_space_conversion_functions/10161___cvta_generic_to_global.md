# 10.16.1. __cvta_generic_to_global()

### 10.16.1. \_\_cvta\_generic\_to\_global()[](#cvta-generic-to-global "Permalink to this headline")

```
__device__ size_t __cvta_generic_to_global(const void *ptr);
```

Returns the result of executing the *PTX*`cvta.to.global` instruction on the generic address denoted by `ptr`.