# 9.7.9.25. Data Movement and Conversion Instructions: Asynchronous copy

#### 9.7.9.25. [Data Movement and Conversion Instructions: Asynchronous copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy "Permalink to this headline")

An asynchronous copy operation performs the underlying operation asynchronously in the background,
thus allowing the issuing threads to perform subsequent tasks.

An asynchronous copy operation can be a *bulk* operation that operates on a large amount of data, or
a *non-bulk* operation that operates on smaller sized data. The amount of data handled by a bulk
asynchronous operation must be a multiple of 16 bytes.

An asynchronous copy operation typically includes the following sequence:

* Optionally, reading from the tensormap.
* Reading data from the source location(s).
* Writing data to the destination location(s).
* Writes being made visible to the executing thread or other threads.