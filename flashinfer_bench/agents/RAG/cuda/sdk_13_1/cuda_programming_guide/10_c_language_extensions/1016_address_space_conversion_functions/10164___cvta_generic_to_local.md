# 10.16.4. __cvta_generic_to_local()

### 10.16.4. \_\_cvta\_generic\_to\_local()[](#cvta-generic-to-local "Permalink to this headline")

```
__device__ size_t __cvta_generic_to_local(const void *ptr);
```

Returns the result of executing the *PTX*`cvta.to.local` instruction on the generic address denoted by `ptr`.