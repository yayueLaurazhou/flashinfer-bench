# enum CUmemDecompressAlgorithm

Bitmasks for CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK.

###### Values

**CU_MEM_DECOMPRESS_UNSUPPORTED = 0**
Decompression is unsupported.
**CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE = 1<<0**
Deflate is supported.
**CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY = 1<<1**
Snappy is supported.
**CU_MEM_DECOMPRESS_ALGORITHM_LZ4 = 1<<2**
LZ4 is supported.