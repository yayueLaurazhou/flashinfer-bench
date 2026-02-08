# enum cudaMemcpyKind

CUDA memory copy types

##### Values

**cudaMemcpyHostToHost = 0**
Host -> Host
**cudaMemcpyHostToDevice = 1**
Host -> Device
**cudaMemcpyDeviceToHost = 2**
Device -> Host
**cudaMemcpyDeviceToDevice = 3**
Device -> Device
**cudaMemcpyDefault = 4**
Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing