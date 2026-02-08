# 13.6.2.1.1. Device-Side Kernel Launch (CDP1)

##### 13.6.2.1.1. Device-Side Kernel Launch (CDP1)[](#device-side-kernel-launch-cdp1 "Permalink to this headline")

See [Kernel Launch APIs](#id237), above, for CDP2 version of document.

Kernels may be launched from the device using the standard CUDA <<< >>> syntax:

```
kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);
```

* `Dg` is of type `dim3` and specifies the dimensions and size of the grid
* `Db` is of type `dim3` and specifies the dimensions and size of each thread block
* `Ns` is of type `size_t` and specifies the number of bytes of shared memory that is dynamically allocated per thread block for this call and addition to statically allocated memory. `Ns` is an optional argument that defaults to 0.
* `S` is of type `cudaStream_t` and specifies the stream associated with this call. The stream must have been allocated in the same thread block where the call is being made. `S` is an optional argument that defaults to 0.