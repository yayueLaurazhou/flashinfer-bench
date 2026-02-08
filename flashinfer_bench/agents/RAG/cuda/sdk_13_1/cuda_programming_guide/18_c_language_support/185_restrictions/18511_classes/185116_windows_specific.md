# 18.5.11.6. Windows-Specific

#### 18.5.11.6. Windows-Specific[](#windows-specific "Permalink to this headline")

The CUDA compiler follows the IA64 ABI for class layout, while the Microsoft host compiler does not. Let `T` denote a pointer to member type, or a class type that satisfies any of the following conditions:

* `T` has virtual functions.
* `T` has a virtual base class.
* `T` has multiple inheritance with more than one direct or indirect empty base class.
* All direct and indirect base classes `B` of `T` are empty and the type of the first field `F` of `T` uses `B` in its definition, such that `B` is laid out at offset 0 in the definition of `F`.

Let `C` denote `T` or a class type that has `T` as a field type or as a base class type. The CUDA compiler may compute the class layout and size differently than the Microsoft host compiler for the type `C`.

As long as the type `C` is used exclusively in host or device code, the program should work correctly.

Passing an object of type `C` between host and device code has undefined behavior, for example, as an argument to a `__global__` function or through `cudaMemcpy*()` calls.

Accessing an object of type `C` or any subobject in device code, or invoking a member function in device code, has undefined behavior if the object is created in host code.

Accessing an object of type `C` or any subobject in host code, or invoking a member function in host code, has undefined behavior if the object is created in device code [12](#fn19).