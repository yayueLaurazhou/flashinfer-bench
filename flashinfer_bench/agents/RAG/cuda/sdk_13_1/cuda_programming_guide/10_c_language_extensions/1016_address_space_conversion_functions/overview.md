# 10.16. Address Space Conversion Functions

## 10.16. Address Space Conversion Functions[](#address-space-conversion-functions "Permalink to this headline")

CUDA C++ pointers (`T*`) can access CUDA C++ objects independently of where these objects are stored.
For example, an `int*` can access `int` objects independently of whether they are stored in global or shared memory.

The Address Space Conversion Functions below enable converting CUDA C++ pointers from and to other representations.
This is required, among others, to interoperate with certain PTX instructions, or to exploit properties of these other representations for performance optimizations.

As an example of interoperating with certain PTX instructions, an `ld.shared.u32 r0, [addr];` PTX instruction expects `addr` to refer to the `shared` space.
A CUDA C++ program with a CUDA C++ `uint32_t*` pointer to an object in `__shared__` memory, needs to convert this pointer to the `shared` space before passing it to such a PTX instruction by calling `__cvta_generic_to_shared` as follows:

```
__shared__ uint32_t x;
x = 42;
void* p = &x;
size_t sp = __cvta_generic_to_shared(p);
uint32_t o;
asm volatile("ld.shared.u32 %0, [%1];" : "=r"(o) : "l"(sp) : "memory");
assert(o == 42);
```

A common program optimization that exploits properties of these other address representations is reducing data-structure size by leveraging that the address ranges of shared, local, and const spaces is smaller than 32-bit, which allows programs to store 32-bit addresses instead of 64-bit pointers.
To obtain the 32-bit integer representation of these addresses, it suffices to truncate it by performing an unsigned 64-bit integer to unsigned 32-bit integer cast:

```
__shared__ int x;
void* p = &x;
uint32_t smem32 = __cvta_generic_to_shared(p);
```

To obtain a generic address from such a 32-bit integer representation, it suffices to zero-extend the address back to an unsigned 64-bit integer before calling the corresponding address space conversion function:

```
size_t smem64 = smem32;
void* q = __cvta_shared_to_generic(smem64);
assert(p == q);
```

A roundtrip from an input generic space pointer to its 32-bit integer representation and back to an output generic space pointer is guaranteed to return an output pointer that is equivalent to the input pointer of the roundtrip for the spaces listed above.