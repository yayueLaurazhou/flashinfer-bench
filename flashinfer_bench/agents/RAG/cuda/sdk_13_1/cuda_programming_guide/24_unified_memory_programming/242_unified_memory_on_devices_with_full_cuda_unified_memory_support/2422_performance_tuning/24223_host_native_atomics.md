# 24.2.2.3. Host Native Atomics

#### 24.2.2.3. Host Native Atomics[](#host-native-atomics "Permalink to this headline")

Some devices, including NVLink-connected devices in
[hardware coherent systems](#um-hw-coherency), support hardware-accelerated
atomic accesses to CPU-resident memory. This implies that atomic accesses to host memory
do not have to be emulated with a page fault.
For these devices, the attribute `cudaDevAttrHostNativeAtomicSupported` is set to 1.