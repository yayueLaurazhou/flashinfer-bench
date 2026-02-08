# 14.9.4. Bind Memory to Multicast Objects

### 14.9.4. Bind Memory to Multicast Objects[](#bind-memory-to-multicast-objects "Permalink to this headline")

After a Multicast Object has been created and all participating devices have been added to the Multicast Object it needs to be backed with
physical memory allocated with `cuMemCreate` for each device:

```
cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, 0 /*flags*/);
```