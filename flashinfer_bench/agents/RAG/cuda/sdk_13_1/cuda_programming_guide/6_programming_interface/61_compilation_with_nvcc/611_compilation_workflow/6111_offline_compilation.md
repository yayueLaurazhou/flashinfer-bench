# 6.1.1.1. Offline Compilation

#### 6.1.1.1. Offline Compilation[](#offline-compilation "Permalink to this headline")

Source files compiled with `nvcc` can include a mix of host code (i.e., code that executes on the host) and device code (i.e., code that executes on the device). `nvcc`’s basic workflow consists in separating device code from host code and then:

* compiling the device code into an assembly form (*PTX* code) and/or binary form (*cubin* object),
* and modifying the host code by replacing the `<<<...>>>` syntax introduced in [Kernels](#kernels) (and described in more details in [Execution Configuration](#execution-configuration)) by the necessary CUDA runtime function calls to load and launch each compiled kernel from the *PTX* code and/or *cubin* object.

The modified host code is output either as C++ code that is left to be compiled using another tool or as object code directly by letting `nvcc` invoke the host compiler during the last compilation stage.

Applications can then:

* Either link to the compiled host code (this is the most common case),
* Or ignore the modified host code (if any) and use the CUDA driver API (see [Driver API](#driver-api)) to load and execute the *PTX* code or *cubin* object.