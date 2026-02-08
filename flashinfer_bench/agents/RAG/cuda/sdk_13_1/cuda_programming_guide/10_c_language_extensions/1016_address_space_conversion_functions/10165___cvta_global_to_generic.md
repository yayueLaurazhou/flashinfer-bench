# 10.16.5. __cvta_global_to_generic()

### 10.16.5. \_\_cvta\_global\_to\_generic()[](#cvta-global-to-generic "Permalink to this headline")

```
__device__ void * __cvta_global_to_generic(size_t rawbits);
```

Returns the generic pointer obtained by executing the *PTX*`cvta.global` instruction on the value provided by `rawbits`.