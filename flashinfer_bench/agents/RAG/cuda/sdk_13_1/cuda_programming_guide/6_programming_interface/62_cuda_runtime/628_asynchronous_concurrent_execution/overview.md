# 6.2.8. Asynchronous Concurrent Execution

### 6.2.8. Asynchronous Concurrent Execution[](#asynchronous-concurrent-execution "Permalink to this headline")

CUDA exposes the following operations as independent tasks that can operate concurrently with one another:

* Computation on the host;
* Computation on the device;
* Memory transfers from the host to the device;
* Memory transfers from the device to the host;
* Memory transfers within the memory of a given device;
* Memory transfers among devices.

The level of concurrency achieved between these operations will depend on the feature set and compute capability of the device as described below.