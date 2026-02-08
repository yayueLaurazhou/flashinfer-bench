# 15.13.5. CPU Virtual Memory

### 15.13.5. CPU Virtual Memory[](#cpu-virtual-memory "Permalink to this headline")

When using CUDA stream-ordered memory allocator APIs, avoid setting VRAM limitations with “ulimit -v” as this is not supported.