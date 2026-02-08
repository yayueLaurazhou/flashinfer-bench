# 18.7.3. Notes on __host__ __device__ lambdas

### 18.7.3. Notes on \_\_host\_\_ \_\_device\_\_ lambdas[](#notes-on-host-device-lambdas "Permalink to this headline")

Unlike `__device__` lambdas, `__host__ __device__` lambdas can be called from host code. As described earlier, the CUDA compiler replaces an extended lambda expression defined in host code with an instance of a named placeholder type. The placeholder type for an extended `__host__ __device__` lambda invokes the original lambda’s `operator()` with an indirect function call [24](#fn31).

The presence of the indirect function call may cause an extended `__host__ __device__` lambda to be less optimized by the host compiler than lambdas that are implicitly or explicitly `__host__` only. In the latter case, the host compiler can easily inline the body of the lambda into the calling context. But in case of an extended `__host__                                  __device__` lambda, the host compiler encounters the indirect function call and may not be able to easily inline the original `__host__ __device__` lambda body.