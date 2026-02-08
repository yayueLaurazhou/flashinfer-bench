# 15.11. IPC Memory Pools

## 15.11. IPC Memory Pools[](#ipc-memory-pools "Permalink to this headline")

IPC capable memory pools allow easy, efficient and secure sharing of GPU memory between processes. CUDA’s IPC memory pools provide the same security benefits as CUDA’s virtual memory management APIs.

There are two phases to sharing memory between processes with memory pools. The processes first need to share access to the pool, then share specific allocations from that pool. The first phase establishes and enforces security. The second phase coordinates what virtual addresses are used in each process and when mappings need to be valid in the importing process.