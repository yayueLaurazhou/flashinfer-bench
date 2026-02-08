# 18.5.2.1. __CUDA_ARCH__

#### 18.5.2.1. \_\_CUDA\_ARCH\_\_[](#cuda-arch "Permalink to this headline")

1. The type signature of the following entities shall not depend on whether `__CUDA_ARCH__` is defined or not, or on a particular value of `__CUDA_ARCH__`:

   * `__global__` functions and function templates
   * `__device__` and `__constant__` variables
   * textures and surfaces

   Example:

   ```
   #if !defined(__CUDA_ARCH__)
   typedef int mytype;
   #else
   typedef double mytype;
   #endif

   __device__ mytype xxx;         // error: xxx's type depends on __CUDA_ARCH__
   __global__ void foo(mytype in, // error: foo's type depends on __CUDA_ARCH__
                       mytype *ptr)
   {
     *ptr = in;
   }
   ```
2. If a `__global__` function template is instantiated and launched from the host, then the function template must be instantiated with the same template arguments irrespective of whether `__CUDA_ARCH__` is defined and regardless of the value of `__CUDA_ARCH__`.

   Example:

   ```
   __device__ int result;
   template <typename T>
   __global__ void kern(T in)
   {
     result = in;
   }

   __host__ __device__ void foo(void)
   {
   #if !defined(__CUDA_ARCH__)
     kern<<<1,1>>>(1);      // error: "kern<int>" instantiation only
                            // when __CUDA_ARCH__ is undefined!
   #endif
   }

   int main(void)
   {
     foo();
     cudaDeviceSynchronize();
     return 0;
   }
   ```
3. In separate compilation mode, the presence or absence of a definition of a function or variable with external linkage shall not depend on whether `__CUDA_ARCH__` is defined or on a particular value of `__CUDA_ARCH__`[7](#fn14).

   Example:

   ```
   #if !defined(__CUDA_ARCH__)
   void foo(void) { }                  // error: The definition of foo()
                                       // is only present when __CUDA_ARCH__
                                       // is undefined
   #endif
   ```
4. In separate compilation, `__CUDA_ARCH__` must not be used in headers such that different objects could contain different behavior. Or, it must be guaranteed that all objects will compile for the same compute\_arch. If a weak function or template function is defined in a header and its behavior depends on `__CUDA_ARCH__`, then the instances of that function in the objects could conflict if the objects are compiled for different compute arch.

   For example, if an a.h contains:

   ```
   template<typename T>
   __device__ T* getptr(void)
   {
   #if __CUDA_ARCH__ == 700
     return NULL; /* no address */
   #else
     __shared__ T arr[256];
     return arr;
   #endif
   }
   ```

   Then if `a.cu` and `b.cu` both include `a.h` and instantiate `getptr` for the same type, and `b.cu` expects a non-NULL address, and compile with:

   ```
   nvcc –arch=compute_70 –dc a.cu
   nvcc –arch=compute_80 –dc b.cu
   nvcc –arch=sm_80 a.o b.o
   ```

   At link time only one version of the `getptr` is used, so the behavior would depend on which version is chosen. To avoid this, either `a.cu` and `b.cu` must be compiled for the same compute arch, or `__CUDA_ARCH__` should not be used in the shared header function.

The compiler does not guarantee that a diagnostic will be generated for the unsupported uses of `__CUDA_ARCH__` described above.