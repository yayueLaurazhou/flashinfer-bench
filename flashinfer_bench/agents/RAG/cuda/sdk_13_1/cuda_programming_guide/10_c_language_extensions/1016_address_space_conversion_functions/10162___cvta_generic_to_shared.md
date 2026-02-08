# 10.16.2. __cvta_generic_to_shared()

### 10.16.2. \_\_cvta\_generic\_to\_shared()[](#cvta-generic-to-shared "Permalink to this headline")

```
__device__ size_t __cvta_generic_to_shared(const void *ptr);
```

Returns the result of executing the *PTX*`cvta.to.shared` instruction on the generic address denoted by `ptr`.