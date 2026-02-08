# CUresult cuLogsCurrent (CUlogIterator *iterator_out, unsigned int flags)

Sets log iterator to point to the end of log buffer, where the next message would be written.

###### Parameters

**iterator_out**

  - Location to store an iterator to the current tail of the logs


CUDA Driver API TRM-06703-001 _vRelease Version  |  590


Modules


**flags**

  - Reserved for future use, must be 0

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE