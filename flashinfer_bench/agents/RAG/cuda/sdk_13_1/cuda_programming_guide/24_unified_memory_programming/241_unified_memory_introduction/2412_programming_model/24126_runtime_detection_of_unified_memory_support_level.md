# 24.1.2.6. Runtime detection of Unified Memory Support Level

#### 24.1.2.6. Runtime detection of Unified Memory Support Level[](#runtime-detection-of-unified-memory-support-level "Permalink to this headline")

The following example shows how to detect the Unified Memory support level at runtime:

```
int main() {
  int d;
  cudaGetDevice(&d);

  int pma = 0;
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, d);
  printf("Full Unified Memory Support: %s\n", pma == 1? "YES" : "NO");
  
  int cma = 0;
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, d);
  printf("CUDA Managed Memory with full support: %s\n", cma == 1? "YES" : "NO");

  return 0;
}
```