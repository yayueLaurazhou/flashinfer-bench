# 10.27.6.2. Trivially copyable

#### 10.27.6.2. Trivially copyable[](#trivially-copyable "Permalink to this headline")

On devices with compute capability 8.0, the [cp.async family of instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) allows copying data from global to shared memory asynchronously. If the pointer types passed to `memcpy_async` do not point to [TriviallyCopyable](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable) types, the copy constructor of each output element needs to be invoked, and these instructions cannot be used to accelerate `memcpy_async`.