# Memcpy

In the reference documentation, each memcpy function is categorized as synchronous or asynchronous,
corresponding to the definitions below.

Synchronous

1. For transfers from pageable host memory to device memory, a stream sync is performed before the
copy is initiated. The function will return once the pageable buffer has been copied to the staging
memory for DMA transfer to device memory, but the DMA to final destination may not have
completed.

2. For transfers from pinned host memory to device memory, the function is synchronous with respect
to the host.

3. For transfers from device to either pageable or pinned host memory, the function returns only once
the copy has completed.

4. For transfers from device memory to device memory, no host-side synchronization is performed.

5. For transfers from any host memory to any host memory, the function is fully synchronous with
respect to the host.

Asynchronous

1. For transfers between device memory and pageable host memory, the function might be
synchronous with respect to host.

2. For transfers from any host memory to any host memory, the function is fully synchronous with
respect to the host.


CUDA Driver API TRM-06703-001 _vRelease Version  |  3


API synchronization behavior


3. If pageable memory must first be staged to pinned memory, the driver may synchronize with the
stream and stage the copy into pinned memory.

4. For all other transfers, the function should be fully asynchronous.