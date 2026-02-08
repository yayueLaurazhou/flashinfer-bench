# 21.5.1. Introduction

### 21.5.1. Introduction[](#introduction-driver-entry-point-access "Permalink to this headline")

The `Driver Entry Point Access APIs` provide a way to retrieve the address of a CUDA driver function. Starting from CUDA 11.3, users can call into available CUDA driver APIs using function pointers obtained from these APIs.

These APIs provide functionality similar to their counterparts, dlsym on POSIX platforms and GetProcAddress on Windows. The provided APIs will let users:

* Retrieve the address of a driver function using the `CUDA Driver API.`
* Retrieve the address of a driver function using the `CUDA Runtime API.`
* Request *per-thread default stream* version of a CUDA driver function. For more details, see [Retrieve Per-thread Default Stream Versions](#retrieve-per-thread-default-stream-versions).
* Access new CUDA features on older toolkits but with a newer driver.