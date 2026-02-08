# 24. Unified Memory Programming

# 24. Unified Memory Programming[](#unified-memory-programming "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

Note

This chapter applies to devices with compute capability 5.0 or higher unless stated otherwise.
For devices with compute capability lower than 5.0, refer to the CUDA toolkit documentation for CUDA 11.8.

This documentation on Unified Memory is divided into 3 parts:

* [General description of unified memory](#um-introduction)
* [Unified Memory on devices with full CUDA Unified Memory support](#um-pageable-systems)
* [Unified Memory on devices without full CUDA Unified Memory support](#um-no-pageable-systems)