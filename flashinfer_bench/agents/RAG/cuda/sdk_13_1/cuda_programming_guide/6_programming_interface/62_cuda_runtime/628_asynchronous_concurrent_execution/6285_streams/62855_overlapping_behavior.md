# 6.2.8.5.5. Overlapping Behavior

##### 6.2.8.5.5. Overlapping Behavior[](#overlapping-behavior "Permalink to this headline")

The amount of execution overlap between two streams depends on the order in which the commands are issued to each stream
and whether or not the device supports overlap of data transfer and kernel execution (see [Overlap of Data Transfer and Kernel Execution](#overlap-of-data-transfer-and-kernel-execution)),
concurrent kernel execution (see [Concurrent Kernel Execution](#concurrent-kernel-execution)), and/or concurrent data transfers (see [Concurrent Data Transfers](#concurrent-data-transfers)).

For example, on devices that do not support concurrent data transfers, the two streams of the code sample of [Creation and Destruction of Streams](#creation-and-destruction-streams)
do not overlap at all because the memory copy from host to device is issued to stream[1] after the memory copy from device to host
is issued to stream[0], so it can only start once the memory copy from device to host issued to stream[0] has completed. If the
code is rewritten the following way (and assuming the device supports overlap of data transfer and kernel execution)

```
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
```

then the memory copy from host to device issued to stream[1] overlaps with the kernel launch issued to stream[0].

On devices that do support concurrent data transfers, the two streams of the code sample of [Creation and Destruction of Streams](#creation-and-destruction-streams)
do overlap: The memory copy from host to device issued to stream[1] overlaps with the memory copy from device to host issued to
stream[0] and even with the kernel launch issued to stream[0] (assuming the device supports overlap of data transfer and kernel execution).