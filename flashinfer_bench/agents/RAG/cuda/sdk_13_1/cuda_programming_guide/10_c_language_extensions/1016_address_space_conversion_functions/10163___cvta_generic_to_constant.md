# 10.16.3. __cvta_generic_to_constant()

### 10.16.3. \_\_cvta\_generic\_to\_constant()[](#cvta-generic-to-constant "Permalink to this headline")

```
__device__ size_t __cvta_generic_to_constant(const void *ptr);
```

Returns the result of executing the *PTX*`cvta.to.const` instruction on the generic address denoted by `ptr`.