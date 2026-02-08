# 13.6.1.2.1.6. Texture Memory (CDP1)

###### 13.6.1.2.1.6. Texture Memory (CDP1)[](#texture-memory-cdp1 "Permalink to this headline")

See [Texture Memory](#texture-memory-cdp), above, for CDP2 version of document.

Writes to the global memory region over which a texture is mapped are incoherent with respect to texture accesses. Coherence for texture memory is enforced at the invocation of a child grid and when a child grid completes. This means that writes to memory prior to a child kernel launch are reflected in texture memory accesses of the child. Similarly, writes to memory by a child will be reflected in the texture memory accesses by a parent, but only after the parent synchronizes on the child’s completion. Concurrent accesses by parent and child may result in inconsistent data.

Warning

Explicit synchronization with child kernels from a parent block (i.e. using `cudaDeviceSynchronize()` in device code) is deprecated in CUDA 11.6, removed for compute\_90+ compilation, and is slated for full removal in a future CUDA release.