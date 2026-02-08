# 10.16.6. __cvta_shared_to_generic()

### 10.16.6. \_\_cvta\_shared\_to\_generic()[](#cvta-shared-to-generic "Permalink to this headline")

```
__device__ void * __cvta_shared_to_generic(size_t rawbits);
```

Returns the generic pointer obtained by executing the *PTX*`cvta.shared` instruction on the value provided by `rawbits`.