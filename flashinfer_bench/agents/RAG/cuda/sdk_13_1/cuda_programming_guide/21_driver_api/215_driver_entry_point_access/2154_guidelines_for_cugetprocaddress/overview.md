# 21.5.4. Guidelines for cuGetProcAddress

### 21.5.4. Guidelines for cuGetProcAddress[](#guidelines-for-cugetprocaddress "Permalink to this headline")

Below are guidelines to keep in mind when using `cuGetProcAddress`.

* Code the CUDA version passed to `cuGetProcAddress` to match the typedef version (do not use a compile time constant such as `CUDA_VERSION` or a dynamic version such as returned from `cuDriverGetVersion`)
* Check the current driver version (such as from `cuDriverGetVersion`) is sufficient before calling `cuGetProcAddress` or an error is expected or an unexpected symbol may be returned