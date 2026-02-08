# 6.2.8.8.2. Elapsed Time

##### 6.2.8.8.2. Elapsed Time[](#elapsed-time "Permalink to this headline")

The events created in [Creation and Destruction of Events](#creation-and-destruction-events) can be used to time the code sample of [Creation and Destruction of Streams](#creation-and-destruction-streams) the following way:

```
cudaEventRecord(start, 0);
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>
               (outputDev + i * size, inputDev + i * size, size);
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
```