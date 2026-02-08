# 18.5.22.10. Defaulted functions

#### 18.5.22.10. Defaulted functions[](#defaulted-functions "Permalink to this headline")

Execution space specifiers on a non-virtual function that is explicitly-defaulted on its first declaration are ignored by the CUDA compiler. Instead, the CUDA compiler will infer the execution space specifiers as described in [Implicitly-declared and non-virtual explicitly-defaulted functions](#compiler-generated-functions).

Execution space specifiers are not ignored if the function is either:

* Explicitly-defaulted but not on its first declaration.
* Explicitly-defaulted and virtual.

Example:

```
 struct S1 {
   // warning: __host__ annotation is ignored on a non-virtual function that
   //          is explicitly-defaulted on its first declaration
   __host__ S1() = default;
 };

 __device__ void foo1() {
   //note: __device__ execution space is derived for S1::S1
   //       based on implicit call from within __device__ function
   //       foo1
   S1 s1;
 }

 struct S2 {
   __host__ S2();
 };

 //note: S2::S2 is not defaulted on its first declaration, and
 //      its execution space is fixed to __host__  based on its
 //      first declaration.
 S2::S2() = default;

 __device__ void foo2() {
    // error: call from __device__ function 'foo2' to
    //        __host__ function 'S2::S2'
    S2 s2;
 }

struct S3 {
  //note: S3::~S3 has __host__ execution space
  virtual __host__ ~S3() = default;
};

__device__ void foo3() {
  S3 qqq;
}  /*(implicit destructor call for 'qqq'):
      error: call from a __device__ fuction 'foo3' to a
     __host__ function 'S3::~S3' */
```