# __device__ void cudaGraphSetConditional (cudaGraphConditionalHandle handle, unsigned int value)

Sets the condition value associated with a conditional node.

##### Description

Sets the condition value associated with a conditional node.

Note: handle must be associated with the same context as the kernel calling this function. Note: It is
undefined behavior to have racing / possibly concurrent calls to cudaGraphSetConditional.

See also:

cudaGraphConditionalHandleCreate