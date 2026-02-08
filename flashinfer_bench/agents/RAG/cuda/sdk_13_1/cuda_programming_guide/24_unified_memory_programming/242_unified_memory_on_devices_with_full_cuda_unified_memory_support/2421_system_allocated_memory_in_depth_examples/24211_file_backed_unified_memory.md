# 24.2.1.1. File-backed Unified Memory

#### 24.2.1.1. File-backed Unified Memory[](#file-backed-unified-memory "Permalink to this headline")

Since [systems with full CUDA Unified Memory support](#um-requirements)
allow the device to access any memory owned by the host process,
they can directly access file-backed memory.

Here, we show a modified version of the initial example shown in the previous section to use
file-backed memory in order to print a string from the GPU, read directly from an input file.
In the following example, the memory is backed by a physical file, but the example
applies to memory-backed files, too, as detailed in the section on
[Inter-Process Communication (IPC) with Unified Memory](#um-sam-ipc).

```
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

```
void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "Invalid file handle");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "Invalid file stats");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
  kernel<<<1, 1>>>("file-backed", mapped);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
  ASSERT(close(fd) == 0, "Cannot close file");
}
```

Note that on systems without the `hostNativeAtomicSupported` property, including
[systems with Linux HMM enabled](#um-requirements),
atomic accesses to file-backed memory are not supported.