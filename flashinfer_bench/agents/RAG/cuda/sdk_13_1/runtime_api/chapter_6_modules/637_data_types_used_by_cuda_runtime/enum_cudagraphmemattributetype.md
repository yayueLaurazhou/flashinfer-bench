# enum cudaGraphMemAttributeType

Graph memory attributes

##### Values

**cudaGraphMemAttrUsedMemCurrent = 0x0**
(value type = cuuint64_t) Amount of memory, in bytes, currently associated with graphs.
**cudaGraphMemAttrUsedMemHigh = 0x1**
(value type = cuuint64_t) High watermark of memory, in bytes, associated with graphs since the last
time it was reset. High watermark can only be reset to zero.
**cudaGraphMemAttrReservedMemCurrent = 0x2**
(value type = cuuint64_t) Amount of memory, in bytes, currently allocated for use by the CUDA
graphs asynchronous allocator.
**cudaGraphMemAttrReservedMemHigh = 0x3**
(value type = cuuint64_t) High watermark of memory, in bytes, currently allocated for use by the
CUDA graphs asynchronous allocator.