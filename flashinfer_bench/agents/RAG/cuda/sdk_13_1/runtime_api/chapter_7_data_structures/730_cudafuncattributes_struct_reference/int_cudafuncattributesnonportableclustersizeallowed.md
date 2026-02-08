# int cudaFuncAttributes::nonPortableClusterSizeAllowed

Whether the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed. A
non-portable cluster size may only function on the specific SKUs the program is tested on. The launch
might fail if the program is run on a different hardware platform.

CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking whether the desired
size can be launched on the current device.

Portable Cluster Size

A portable cluster size is guaranteed to be functional on all compute capabilities higher than the target
compute capability. The portable cluster size for sm_90 is 8 blocks per cluster. This value may increase
for future compute capabilities.

The specific hardware unit may support higher cluster sizes that’s not guaranteed to be portable. See
cudaFuncSetAttribute