# 10.16.8. __cvta_local_to_generic()

### 10.16.8. \_\_cvta\_local\_to\_generic()[](#cvta-local-to-generic "Permalink to this headline")

```
__device__ void * __cvta_local_to_generic(size_t rawbits);
```

Returns the generic pointer obtained by executing the *PTX*`cvta.local` instruction on the value provided by `rawbits`.