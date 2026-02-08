# 25.5.2. Allocators

### 25.5.2. Allocators[](#allocators "Permalink to this headline")

Lazy Loading delays loading code from initialization phase of the program closer to execution phase.
Loading code onto the GPU requires memory allocation.

If your application tries to allocate the entire VRAM on startup, for example, to use it for its own allocator,
then it might turn out that there will be no more memory left to load the kernels.
This is despite the fact that overall Lazy Loading frees up more memory for the user.
CUDA will need to allocate some memory to load each kernel, which usually happens at first launch time of each kernel.
If your application allocator greedily allocated everything, CUDA will fail to allocate memory.

Possible solutions:

* use `cudaMallocAsync()` instead of an allocator that allocates the entire VRAM on startup
* add some buffer to compensate for the delayed loading of kernels
* preload all kernels that will be used in the program before trying to initialize your allocator