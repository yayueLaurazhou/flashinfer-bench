# 10.2.4. __grid_constant__

### 10.2.4. \_\_grid\_constant\_\_[](#grid-constant "Permalink to this headline")

The `__grid_constant__` annotation for compute architectures greater or equal to 7.0 annotates a `const`-qualified `__global__` function parameter of non-reference type that:

* Has the lifetime of the grid,
* Is private to the grid, i.e., the object is not accessible to host threads and threads from other grids, including sub-grids,
* Has a distinct object per grid, i.e., all threads in the grid see the same address,
* Is read-only, i.e., modifying a `__grid_constant__` object or any of its sub-objects is *undefined behavior*, including `mutable` members.

Requirements:

* Kernel parameters annotated with `__grid_constant__` must have `const`-qualified non-reference types.
* All function declarations must match with respect to any `__grid_constant_` parameters.
* A function template specialization must match the primary template declaration with respect to any `__grid_constant__` parameters.
* A function template instantiation directive must match the primary template declaration with respect to any `__grid_constant__` parameters.

If the address of a `__global__` function parameter is taken, the compiler will ordinarily make a copy of the kernel parameter in thread local memory and use the address of the copy, to partially support C++ semantics, which allow each thread to modify its own local copy of function parameters. Annotating a `__global__` function parameter with `__grid_constant__` ensures that the compiler will not create a copy of the kernel parameter in thread local memory, but will instead use the generic address of the parameter itself. Avoiding the local copy may result in improved performance.

```
__device__ void unknown_function(S const&);
__global__ void kernel(const __grid_constant__ S s) {
   s.x += threadIdx.x;  // Undefined Behavior: tried to modify read-only memory

   // Compiler will _not_ create a per-thread thread local copy of "s":
   unknown_function(s);
}
```