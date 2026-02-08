# enum cudaMemLocationType

Specifies the type of location


CUDA Runtime API vRelease Version  |  571


Modules

##### Values

**cudaMemLocationTypeInvalid = 0**
**cudaMemLocationTypeNone = 0**
Location is unspecified. This is used when creating a managed memory pool to indicate no preferred
location for the pool
**cudaMemLocationTypeDevice = 1**
Location is a device location, thus id is a device ordinal
**cudaMemLocationTypeHost = 2**
Location is host, id is ignored
**cudaMemLocationTypeHostNuma = 3**
Location is a host NUMA node, thus id is a host NUMA node id
**cudaMemLocationTypeHostNumaCurrent = 4**
Location is the host NUMA node closest to the current thread's CPU, id is ignored