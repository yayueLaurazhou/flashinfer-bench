# 6.2.8.5.4. Implicit Synchronization

##### 6.2.8.5.4. Implicit Synchronization[](#implicit-synchronization "Permalink to this headline")

Two operations from different streams cannot run concurrently if any CUDA operation on the NULL stream is submitted
in-between them, unless the streams are non-blocking streams (created with the `cudaStreamNonBlocking` flag).

Applications should follow these guidelines to improve
their potential for concurrent kernel execution:

* All independent operations should be issued before dependent operations,
* Synchronization of any kind should be delayed as long as possible.