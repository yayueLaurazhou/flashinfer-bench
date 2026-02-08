# enum cudaMemcpyFlags

Flags to specify for copies within a batch. For more details see cudaMemcpyBatchAsync.

##### Values

**cudaMemcpyFlagDefault = 0x0**
**cudaMemcpyFlagPreferOverlapWithCompute = 0x1**
Hint to the driver to try and overlap the copy with compute work on the SMs.