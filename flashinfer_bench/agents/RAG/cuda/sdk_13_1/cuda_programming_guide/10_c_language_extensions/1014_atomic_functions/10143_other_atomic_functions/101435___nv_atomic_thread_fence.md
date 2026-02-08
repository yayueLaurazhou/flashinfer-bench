# 10.14.3.5. __nv_atomic_thread_fence()

#### 10.14.3.5. \_\_nv\_atomic\_thread\_fence()[](#nv-atomic-thread-fence "Permalink to this headline")

```
__device__ void __nv_atomic_thread_fence (int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

This atomic function establishes an ordering between memory accesses requested by this thread based on the specified memory order. And the thread scope parameter specifies the set of threads that may observe the ordering effect of this operation.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.