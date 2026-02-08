# 10.27. Asynchronous Data Copies

## 10.27. Asynchronous Data Copies[](#asynchronous-data-copies "Permalink to this headline")

CUDA 11 introduces Asynchronous Data operations with `memcpy_async` API to allow device code to explicitly manage the asynchronous copying of data. The `memcpy_async` feature enables CUDA kernels to overlap computation with data movement.