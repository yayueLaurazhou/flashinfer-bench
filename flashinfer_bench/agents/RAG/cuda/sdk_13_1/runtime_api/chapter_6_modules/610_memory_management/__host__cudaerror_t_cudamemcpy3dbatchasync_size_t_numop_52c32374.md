# __host__cudaError_t cudaMemcpy3DBatchAsync (size_t numOps, cudaMemcpy3DBatchOp *opList, unsigned long long flags, cudaStream_t stream)

Performs a batch of 3D memory copies asynchronously.

##### Parameters

**numOps**

  - Total number of memcpy operations.
**opList**

  - Array of size numOps containing the actual memcpy operations.
**flags**

  - Flags for future use, must be zero now.


CUDA Runtime API vRelease Version  |  167


Modules


**stream**

##### Returns

cudaSuccess cudaErrorInvalidValue

##### Description

Performs a batch of memory copies. The batch as a whole executes in stream order but copies within
a batch are not guaranteed to execute in any specific order. Note that this means specifying any
dependent copies within a batch will result in undefined behavior.

Performs memory copies as specified in the opList array. The length of this array is specified in
numOps. Each entry in this array describes a copy operation. This includes among other things,
the source and destination operands for the copy as specified in cudaMemcpy3DBatchOp::src
and cudaMemcpy3DBatchOp::dst respectively. The source and destination operands of a copy
can either be a pointer or a CUDA array. The width, height and depth of a copy is specified in
cudaMemcpy3DBatchOp::extent. The width, height and depth of a copy are specified in elements and
must not be zero. For pointer-to-pointer copies, the element size is considered to be 1. For pointer to
CUDA array or vice versa copies, the element size is determined by the CUDA array. For CUDA array
to CUDA array copies, the element size of the two CUDA arrays must match.

For a given operand, if cudaMemcpy3DOperand::type is specified as
cudaMemcpyOperandTypePointer, then cudaMemcpy3DOperand::op::ptr will be used. The
cudaMemcpy3DOperand::op::ptr::ptr field must contain the pointer where the copy should
begin. The cudaMemcpy3DOperand::op::ptr::rowLength field specifies the length of each row in
elements and must either be zero or be greater than or equal to the width of the copy specified in
cudaMemcpy3DBatchOp::extent::width. The cudaMemcpy3DOperand::op::ptr::layerHeight field
specifies the height of each layer and must either be zero or be greater than or equal to the height
of the copy specified in cudaMemcpy3DBatchOp::extent::height. When either of these values is
zero, that aspect of the operand is considered to be tightly packed according to the copy extent. For
managed memory pointers on devices where cudaDevAttrConcurrentManagedAccess is true or
system-allocated pageable memory on devices where cudaDevAttrPageableMemoryAccess is true, the
cudaMemcpy3DOperand::op::ptr::locHint field can be used to hint the location of the operand.

If an operand's type is specified as cudaMemcpyOperandTypeArray, then
cudaMemcpy3DOperand::op::array will be used. The cudaMemcpy3DOperand::op::array::array field
specifies the CUDA array and cudaMemcpy3DOperand::op::array::offset specifies the 3D offset into
that array where the copy begins.

The cudaMemcpyAttributes::srcAccessOrder indicates the source access ordering to
be observed for copies associated with the attribute. If the source access order is set to
cudaMemcpySrcAccessOrderStream, then the source will be accessed in stream order. If the source
access order is set to cudaMemcpySrcAccessOrderDuringApiCall then it indicates that access to
the source pointer can be out of stream order and all accesses must be complete before the API call
returns. This flag is suited for ephemeral sources (ex., stack variables) when it's known that no prior
operations in the stream can be accessing the memory and also that the lifetime of the memory is


CUDA Runtime API vRelease Version  |  168


Modules


limited to the scope that the source variable was declared in. Specifying this flag allows the driver to
optimize the copy and removes the need for the user to synchronize the stream after the API call. If
the source access order is set to cudaMemcpySrcAccessOrderAny then it indicates that access to the
source pointer can be out of stream order and the accesses can happen even after the API call returns.
This flag is suited for host pointers allocated outside CUDA (ex., via malloc) when it's known that
no prior operations in the stream can be accessing the memory. Specifying this flag allows the driver
to optimize the copy on certain platforms. Each memcopy operation in opList must have a valid
srcAccessOrder setting, otherwise this API will return cudaErrorInvalidValue.

The cudaMemcpyAttributes::flags field can be used to specify certain flags for copies. Setting
the cudaMemcpyFlagPreferOverlapWithCompute flag indicates that the associated copies should
preferably overlap with any compute work. Note that this flag is a hint and can be ignored depending
on the platform and other parameters of the copy.