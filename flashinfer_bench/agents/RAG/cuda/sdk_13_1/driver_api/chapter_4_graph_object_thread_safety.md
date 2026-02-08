# Chapter 4. Graph object thread safety

Graph objects (cudaGraph_t, CUgraph) are not internally synchronized and must not be accessed
concurrently from multiple threads. API calls accessing the same graph object must be serialized
externally.

Note that this includes APIs which may appear to be read-only, such as cudaGraphClone()
(cuGraphClone()) and cudaGraphInstantiate() (cuGraphInstantiate()). No API or pair
of APIs is guaranteed to be safe to call on the same graph object from two different threads without
serialization.


CUDA Driver API TRM-06703-001 _vRelease Version  |  7