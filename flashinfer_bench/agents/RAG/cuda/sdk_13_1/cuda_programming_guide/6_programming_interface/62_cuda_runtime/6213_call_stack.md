# 6.2.13. Call Stack

### 6.2.13. Call Stack[](#call-stack "Permalink to this headline")

On devices of compute capability 2.x and higher, the size of the call stack can be queried using`cudaDeviceGetLimit()` and set using `cudaDeviceSetLimit()`.

When the call stack overflows, the kernel call fails with a stack overflow error if the application is run via a CUDA debugger (CUDA-GDB, Nsight) or an unspecified launch error, otherwise.
When the compiler cannot determine the stack size, it issues a warning saying Stack size cannot be statically determined. This is usually the case with recursive functions.
Once this warning is issued, user will need to set stack size manually if default stack size is not sufficient.