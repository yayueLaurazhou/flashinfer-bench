# size_t CUmemDecompressParams::dstNumBytes

The number of bytes that the decompression operation will be expected to write to
CUmemDecompressParams_st.dst. This value is optional; if present, it may be used by the CUDA
driver as a heuristic for scheduling the individual decompression operations.