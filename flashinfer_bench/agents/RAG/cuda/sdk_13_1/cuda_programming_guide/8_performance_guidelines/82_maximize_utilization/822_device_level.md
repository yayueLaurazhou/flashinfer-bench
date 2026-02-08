# 8.2.2. Device Level

### 8.2.2. Device Level[](#device-level "Permalink to this headline")

At a lower level, the application should maximize parallel execution between the multiprocessors of a device.

Multiple kernels can execute concurrently on a device, so maximum utilization can also be achieved by using streams to enable enough kernels to execute concurrently as described in [Asynchronous Concurrent Execution](#asynchronous-concurrent-execution).