# 26.1.3. Allocators and EGM support

### 26.1.3. Allocators and EGM support[](#allocators-and-egm-support "Permalink to this headline")

Mapping system memory as EGM does not cause any performance issues. In
fact, accessing a remote socket’s system memory mapped as EGM is going
to be faster. Because, with EGM traffic is guaranteed to be routed over
NVLinks. Currently, `cuMemCreate` and `cudaMemPoolCreate` allocators are
supported with appropriate location type and NUMA identifiers.