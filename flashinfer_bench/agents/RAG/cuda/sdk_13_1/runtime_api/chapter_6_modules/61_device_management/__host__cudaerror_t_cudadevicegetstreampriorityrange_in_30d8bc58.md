# __host__cudaError_t cudaDeviceGetStreamPriorityRange (int *leastPriority, int *greatestPriority)

Returns numerical values that correspond to the least and greatest stream priorities.

##### Parameters

**leastPriority**

  - Pointer to an int in which the numerical value for least stream priority is returned
**greatestPriority**

  - Pointer to an int in which the numerical value for greatest stream priority is returned

##### Returns

cudaSuccess

##### Description

Returns in *leastPriority and *greatestPriority the numerical values that correspond
to the least and greatest stream priorities respectively. Stream priorities follow a convention where
lower numbers imply greater priorities. The range of meaningful stream priorities is given by

[*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority
value that is outside the the meaningful range as specified by this API, the priority is automatically
clamped down or up to either *leastPriority or *greatestPriority respectively. See
cudaStreamCreateWithPriority for details on creating a priority stream. A NULL may be passed in for
*leastPriority or *greatestPriority if the value is not desired.


CUDA Runtime API vRelease Version  |  22


Modules


This function will return '0' in both *leastPriority and *greatestPriority if the current
context's device does not support stream priorities (see cudaDeviceGetAttribute).



See also:

cudaStreamCreateWithPriority, cudaStreamGetPriority, cuCtxGetStreamPriorityRange