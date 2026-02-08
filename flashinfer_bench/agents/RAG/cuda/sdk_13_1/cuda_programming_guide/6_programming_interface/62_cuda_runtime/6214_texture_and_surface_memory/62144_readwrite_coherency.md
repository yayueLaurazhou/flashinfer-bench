# 6.2.14.4. Read/Write Coherency

#### 6.2.14.4. Read/Write Coherency[](#read-write-coherency "Permalink to this headline")

The texture and surface memory is cached (see [Device Memory Accesses](#device-memory-accesses)) and within the same kernel call, the cache is not kept coherent with respect to global memory writes and surface memory writes, so any texture fetch or surface read to an address that has been written to via a global write or a surface write in the same kernel call returns undefined data. In other words, a thread can safely read some texture or surface memory location only if this memory location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread from the same kernel call.