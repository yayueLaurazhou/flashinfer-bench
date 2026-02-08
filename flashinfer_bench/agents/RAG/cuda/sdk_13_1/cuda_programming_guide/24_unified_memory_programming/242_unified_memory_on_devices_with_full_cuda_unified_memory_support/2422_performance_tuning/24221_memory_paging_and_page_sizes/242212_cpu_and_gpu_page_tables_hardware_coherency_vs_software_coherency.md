# 24.2.2.1.2. CPU and GPU page tables: hardware coherency vs. software coherency

##### 24.2.2.1.2. CPU and GPU page tables: hardware coherency vs. software coherency[](#cpu-and-gpu-page-tables-hardware-coherency-vs-software-coherency "Permalink to this headline")

Note

In the remainder of the performance tuning documentation, we will refer
to systems with a combined page table for both CPUs and GPUs as *hardware
coherent* systems. Systems with separate page tables for CPUs and GPUs are
referred to as *software coherent*.

Hardware coherent systems such as NVIDIA Grace Hopper offer a logically combined
page table for both CPUs and GPUs.
This is important because in order to access
[System-Allocated Memory from the GPU](#um-system-allocator),
the GPU uses whichever page table entry was created by the CPU for the requested memory.
If that page table entry uses the default CPU page size of 4KiB or 64KiB,
accesses to large virtual memory areas will cause significant TLB misses,
thus significant slowdowns.

See the section on configuring huge pages for examples on how to ensure
System-Allocated Memory uses large enough page sizes to avoid this type of issue.

On the other hand, on systems where the CPUs and GPUs each have their own logical
page table, different performance tuning aspects should be considered:
in order to [guarantee coherency](#um-introduction), these systems
usually use *page faults* in case a processor accesses a memory address mapped
into the physical memory of a different processor. Such a page fault means that:

* it needs to be ensured that the currently owning processor (where the physical page currently resides)
  cannot access this page anymore, either by deleting the page table entry or updating it.
* it needs to be ensured that the processor requesting access can access this page,
  either by creating a new page table entry or updating and existing entry, such that
  it becomes valid/active.
* the physical page backing this virtual page must be moved/migrated to the processor
  requesting access: this can be an expensive operation, and the amount of work
  is proportional to the page size.

Overall, hardware coherent systems provide significant performance benefits
compared to software coherent systems in cases where frequent concurrent accesses
to the same memory page are made by both CPU and GPU threads:

* less page-faults: these systems do not need to use page-faults for emulating coherency or migrating memory,
* less contention: these systems are coherent at cache-line granularity instead of page-size granularity, that is,
  when there is contention from multiple processors within a cache line, only the cache line is exchanged which is much smaller than the smallest page-size,
  and when the different processors access different cache-lines within a page, then there is no contention.

This impacts the performance of the following scenarios:

* Atomic updates to the same address concurrently from both CPUs and GPUs.
* Signaling a GPU thread from a CPU thread or vice-versa.