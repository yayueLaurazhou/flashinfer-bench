# 24.2.1.2. Inter-Process Communication (IPC) with Unified Memory

#### 24.2.1.2. Inter-Process Communication (IPC) with Unified Memory[](#inter-process-communication-ipc-with-unified-memory "Permalink to this headline")

Note

As of now, using IPC with Unified Memory can have significant performance implications.

Many applications prefer to manage one GPU per process, but still need to use Unified Memory,
for example for over-subscription, and access it from multiple GPUs.

CUDA IPC (see [Interprocess Communication](#interprocess-communication))
does not support Managed Memory: handles to this type of memory may not be shared through
any of the mechanisms discussed in this section.
On [systems with full CUDA Unified Memory support](#um-requirements),
System-Allocated Memory is Inter-Process Communication (IPC) capable.
Once access to System-Allocated Memory has been shared with other processes,
the same [Programming Model](#um-programming-model) applies,
similar to [File-backed Unified Memory](#um-sam-file-backed).

See the following references for more information on various ways of creating
IPC-capable System-Allocated Memory under Linux:

* [mmap with MAP\_SHARED](https://man7.org/linux/man-pages/man2/mmap.2.html)
* [POSIX IPC APIs](https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html)
* [Linux memfd\_create](https://man7.org/linux/man-pages/man2/memfd_create.2.html)

Note that it is not possible to share memory between different hosts and their devices using this technique.