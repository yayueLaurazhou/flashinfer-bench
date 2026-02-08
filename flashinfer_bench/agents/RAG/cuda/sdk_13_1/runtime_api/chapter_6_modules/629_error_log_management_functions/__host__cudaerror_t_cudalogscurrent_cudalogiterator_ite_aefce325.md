# __host__cudaError_t cudaLogsCurrent (cudaLogIterator *iterator_out, unsigned int flags)

Sets log iterator to point to the end of log buffer, where the next message would be written.

##### Parameters

**iterator_out**

  - Location to store an iterator to the current tail of the logs
**flags**

  - Reserved for future use, must be 0

##### Returns

cudaSuccess, cudaErrorInvalidValue,