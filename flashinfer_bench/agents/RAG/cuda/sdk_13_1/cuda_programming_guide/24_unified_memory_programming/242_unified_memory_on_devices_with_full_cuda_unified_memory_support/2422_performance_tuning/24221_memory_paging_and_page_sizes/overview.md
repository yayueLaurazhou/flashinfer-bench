# 24.2.2.1. Memory Paging and Page Sizes

#### 24.2.2.1. Memory Paging and Page Sizes[](#memory-paging-and-page-sizes "Permalink to this headline")

Many of the sections for unified memory performance tuning assume prior knowledge on virtual addressing,
memory pages and page sizes.
This section attempts to define all necessary terms and explain why paging matters for performance.

All currently supported systems for Unified Memory use a virtual address space:
this means that memory addresses used by an application represent a *virtual* location
which might be *mapped* to a physical location where the memory actually resides.

All currently supported processors, including both CPUs and GPUs, additionally use
memory *paging*. Because all systems use a virtual address space, there are two types
of memory pages:

* Virtual pages: this represents a fixed-size contiguous chunk of virtual memory
  per process tracked by the operating system, which can be *mapped* into physical memory.
  Note that the virtual page is linked to the *mapping*: for example, a single
  virtual address might be mapped into physical memory using different page sizes.
* Physical pages: this represents a fixed-size contiguous chunk of memory
  the processor’s main Memory Management Unit (MMU) supports and into which
  a virtual page can be mapped.

Currently, all x86\_64 CPUs use 4KiB physical pages.
Arm CPUs support multiple physical page sizes - 4KiB, 16KiB, 32KiB and 64KiB - depending on the exact CPU.
Finally, NVIDIA GPUs support multiple physical page sizes, but prefer 2MiB physical pages or larger.
Note that these sizes are subject to change in future hardware.

The default page size of virtual pages usually corresponds to the physical page size,
but an application may use different page sizes as long as they are supported by the
operating system and the hardware. Typically, supported virtual page sizes must be
powers of 2 and multiples of the physical page size.

The logical entity tracking the mapping of virtual pages into physical pages will be referred to as a *page table*,
and each mapping of a given virtual page with a given virtual size to physical pages is called a *page table entry (PTE)*.
All supported processors provide specific caches for the page table to speed up the translation of
virtual addresses to physical addresses. These caches are called *translation lookaside buffers (TLBs)*.

There are two important aspects for performance tuning of applications:

* the choice of virtual page size,
* whether the system offers a combined page table used by both CPUs and GPUs,
  or separate page tables for each CPU and GPU individually.