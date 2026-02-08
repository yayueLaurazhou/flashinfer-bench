# 15.13.2. cuPointerGetAttribute Query

### 15.13.2. cuPointerGetAttribute Query[](#cupointergetattribute-query "Permalink to this headline")

Invoking `cuPointerGetAttribute` on an allocation after invoking `cudaFreeAsync` on it results in undefined behavior. Specifically, it does not matter if an allocation is still accessible from a given stream: the behavior is still undefined.