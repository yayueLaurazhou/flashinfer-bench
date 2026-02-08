# 9.7.13.15.1. Size and alignment of mbarrier object

##### 9.7.13.15.1. [Size and alignment of mbarrier object](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment "Permalink to this headline")

An mbarrier object is an opaque object with the following type and alignment requirements :

| Type | Alignment (bytes) | Memory space |
| --- | --- | --- |
| `.b64` | 8 | `.shared` |