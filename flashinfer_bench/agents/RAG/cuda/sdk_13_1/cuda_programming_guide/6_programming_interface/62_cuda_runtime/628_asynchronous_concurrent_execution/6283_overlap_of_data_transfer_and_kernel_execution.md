# 6.2.8.3. Overlap of Data Transfer and Kernel Execution

#### 6.2.8.3. Overlap of Data Transfer and Kernel Execution[](#overlap-of-data-transfer-and-kernel-execution "Permalink to this headline")

Some devices can perform an asynchronous memory copy to or from the GPU concurrently with kernel execution. Applications may query this capability by checking the `asyncEngineCount` device property (see [Device Enumeration](#device-enumeration)), which is greater than zero for devices that support it. If host memory is involved in the copy, it must be page-locked.

It is also possible to perform an intra-device copy simultaneously with kernel execution (on devices that support the `concurrentKernels` device property) and/or with copies to or from the device (for devices that support the `asyncEngineCount` property). Intra-device copies are initiated using the standard memory copy functions with destination and source addresses residing on the same device.