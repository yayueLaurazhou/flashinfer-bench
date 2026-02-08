# 18.5.3.2. __managed__ Memory Space Specifier

#### 18.5.3.2. \_\_managed\_\_ Memory Space Specifier[](#managed-memory-space-specifier "Permalink to this headline")

Variables marked with the `__managed__` memory space specifier (“managed” variables) have the following restrictions:

* The address of a managed variable is not a constant expression.
* A managed variable shall not have a const qualified type.
* A managed variable shall not have a reference type.
* The address or value of a managed variable shall not be used when the CUDA runtime may not be in a valid state, including the following cases:

  + In static/dynamic initialization or destruction of an object with static or thread local storage duration.
  + In code that executes after exit() has been called (for example, a function marked with gcc’s “`__attribute__((destructor))`”).
  + In code that executes when CUDA runtime may not be initialized (for example, a function marked with gcc’s “`__attribute__((constructor))`”).
* A managed variable cannot be used as an unparenthesized id-expression argument to a `decltype()` expression.
* Managed variables have the same coherence and consistency behavior as specified for dynamically allocated managed memory.
* When a CUDA program containing managed variables is run on an execution platform with multiple GPUs, the variables are allocated only once, and not per GPU.
* A managed variable declaration without the extern linkage is not allowed within a function that executes on the host.
* A managed variable declaration without the extern or static linkage is not allowed within a function that executes on the device.

Here are examples of legal and illegal uses of managed variables:

```
__device__ __managed__ int xxx = 10;         // OK

int *ptr = &xxx;                             // error: use of managed variable
                                             // (xxx) in static initialization
struct S1_t {
  int field;
  S1_t(void) : field(xxx) { };
};
struct S2_t {
  ~S2_t(void) { xxx = 10; }
};

S1_t temp1;                                 // error: use of managed variable
                                            // (xxx) in dynamic initialization

S2_t temp2;                                 // error: use of managed variable
                                            // (xxx) in the destructor of
                                            // object with static storage
                                            // duration

__device__ __managed__ const int yyy = 10;  // error: const qualified type

__device__ __managed__ int &zzz = xxx;      // error: reference type

template <int *addr> struct S3_t { };
S3_t<&xxx> temp;                            // error: address of managed
                                            // variable(xxx) not a
                                            // constant expression

__global__ void kern(int *ptr)
{
  assert(ptr == &xxx);                      // OK
  xxx = 20;                                 // OK
}
int main(void)
{
  int *ptr = &xxx;                          // OK
  kern<<<1,1>>>(ptr);
  cudaDeviceSynchronize();
  xxx++;                                    // OK
  decltype(xxx) qqq;                        // error: managed variable(xxx) used
                                            // as unparenthized argument to
                                            // decltype

  decltype((xxx)) zzz = yyy;                // OK
}
```