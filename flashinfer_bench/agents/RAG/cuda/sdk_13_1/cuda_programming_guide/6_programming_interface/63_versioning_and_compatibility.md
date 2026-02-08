# 6.3. Versioning and Compatibility

## 6.3. Versioning and Compatibility[](#versioning-and-compatibility "Permalink to this headline")

There are two version numbers that developers should care about when developing a CUDA application: The compute capability that describes the general specifications and features of the compute device (see [Compute Capability](#compute-capability)) and the version of the CUDA driver API that describes the features supported by the driver API and runtime.

The version of the driver API is defined in the driver header file as `CUDA_VERSION`. It allows developers to check whether their application requires a newer device driver than the one currently installed. This is important, because the driver API is *backward compatible*, meaning that applications, plug-ins, and libraries (including the CUDA runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases as illustrated in [Figure 26](#versioning-and-compatibility-driver-api-is-backward-but-not-forward-compatible). The driver API is not *forward compatible*, which means that applications, plug-ins, and libraries (including the CUDA runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver.

It is important to note that there are limitations on the mixing and matching of versions that is supported:

* Since only one version of the CUDA Driver can be installed at a time on a system, the installed driver must be of the same or higher version than the maximum Driver API version against which any application, plug-ins, or libraries that must run on that system were built.
* All plug-ins and libraries used by an application must use the same version of the CUDA Runtime unless they statically link to the Runtime, in which case multiple versions of the runtime can coexist in the same process space. Note that if `nvcc` is used to link the application, the static version of the CUDA Runtime library will be used by default, and all CUDA Toolkit libraries are statically linked against the CUDA Runtime.
* All plug-ins and libraries used by an application must use the same version of any libraries that use the runtime (such as cuFFT, cuBLAS, …) unless statically linking to those libraries.

![The Driver API Is Backward but Not Forward Compatible](_images/compatibility-of-cuda-versions.png)


Figure 26 The Driver API Is Backward but Not Forward Compatible[](#versioning-and-compatibility-driver-api-is-backward-but-not-forward-compatible "Permalink to this image")

For Tesla GPU products, CUDA 10 introduced a new forward-compatible upgrade path for the user-mode components of the CUDA Driver. This feature is described in [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html). The requirements on the CUDA Driver version described here apply to the version of the user-mode components.