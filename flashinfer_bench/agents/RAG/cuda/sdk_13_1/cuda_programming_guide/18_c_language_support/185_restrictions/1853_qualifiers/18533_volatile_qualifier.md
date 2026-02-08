# 18.5.3.3. Volatile Qualifier

#### 18.5.3.3. Volatile Qualifier[](#volatile-qualifier "Permalink to this headline")

Note

The `volatile` keyword is supported to maintain compatibility with ISO C++; however, few if any of its [remaining non-deprecated uses](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1152r0.html#prop) apply to GPUs.

Reads and writes to volatile qualified objects are not atomic, and are compiled to one or more [.volatile instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation) which do NOT guarantee:

* ordering of memory operations, or
* that the number of memory operations performed by the HW matches the number of PTX instructions.

That is, CUDA C++ volatile is not suitable for:

* **Inter-Thread Synchronization**: use atomic operations via [cuda::atomic\_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html), or [Atomic Functions](#atomic-functions) instead.
  Atomic memory operations provide inter-thread synchronization guarantees and deliver much better performance than volatile operations.
  CUDA C++ volatile operations do not provide any inter-thread synchronization guarantees and are therefore not correct for inter-thread synchronization.
  The following example shows how to pass a message across two threads using atomic operations.

  > cuda::atomic\_ref
  >
  > |  |
  > | --- |
  > | ``` __global__ void kernel(int* flag, int* data) {   cuda::atomic_ref<int, cuda::thread_scope_device> f{*flag};   if (threadIdx.x == 0) {     // Consumer: blocks until flag is set by producer, then reads data     while(f.load(cuda::memory_order_acquire) == 0);     if (*data != 42) __trap(); // Errors if wrong data read   } else if (threadIdx.x == 1) {     // Producer: writes data then sets flag     *data = 42;     f.store(1, cuda::memory_order_release);   } } ``` |
  >
  >
  > cuda::atomic
  >
  > |  |
  > | --- |
  > | ``` __global__ void kernel(cuda::atomic<int, cuda::thread_scope_device>* flag, int* data) {   if (threadIdx.x == 0) {     // Consumer: blocks until flag is set by producer, then reads data     while(flag->load(cuda::memory_order_acquire) == 0);     if (*data != 42) __trap(); // Errors if wrong data read   } else if (threadIdx.x == 1) {     // Producer: writes data then sets flag     *data = 42;     flag->store(1, cuda::memory_order_release);   } } ``` |
  >
  >
  > Atomic Functions (`atomicAdd` and `atomicExch`)
  >
  > |  |
  > | --- |
  > | ``` __global__ void kernel(int* flag, int* data) {   if (threadIdx.x == 0) {     // Consumer: blocks until flag is set by producer, then reads data     while(atomicAdd(flag, 0) == 0); // Load with Relaxed Read-Modify-Write     __threadfence();                // SequentiallyConsistent fence     if (*data != 42) __trap();      // Errors if wrong data read   } else if (threadIdx.x == 1) {     // Producer: writes data then sets flag     *data = 42;     __threadfence();     // SequentiallyConsistent fence     atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write   } } ``` |
* **Memory Mapped IO** (MMIO): use [PTX MMIO operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation) via inline PTX instead.
  PTX MMIO operations strictly preserve the number of memory accesses performed.
  CUDA C++ `volatile` operations do not preserve the number of memory accesses performed, and may perform more or less accesses than requested in a non-deterministic way, making them incorrect for MMIO.
  The following example shows how to read and write from a register using PTX mmio operations.

  > ```
  > __global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
  >   // Write to MMIO register:
  >   int value = 13;
  >   asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;" :: "l"(mmio_reg0), "r"(value) : "memory");
  >
  >   // Read MMIO register:
  >   asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];" : "=r"(value) : "l"(mmio_reg1) : "memory");
  >   
  >   if (value != 42) __trap(); // Errors if wrong data read
  > }
  > ```