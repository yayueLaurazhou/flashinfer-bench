# 10.16.7. __cvta_constant_to_generic()

### 10.16.7. \_\_cvta\_constant\_to\_generic()[](#cvta-constant-to-generic "Permalink to this headline")

```
__device__ void * __cvta_constant_to_generic(size_t rawbits);
```

Returns the generic pointer obtained by executing the *PTX*`cvta.const` instruction on the value provided by `rawbits`.