# CUresult cuMemcpy3DBatchAsync (size_t numOps, CUDA_MEMCPY3D_BATCH_OP *opList, unsigned long long flags, CUstream hStream)

Performs a batch of 3D memory copies asynchronously.

###### Parameters

**numOps**

  - Total number of memcpy operations.
**opList**

  - Array of size numOps containing the actual memcpy operations.
**flags**

  - Flags for future use, must be zero now.
**hStream**

  - The stream to enqueue the operations in. Must not be default NULL stream.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Performs a batch of memory copies. The batch as a whole executes in stream order but copies within
a batch are not guaranteed to execute in any specific order. Note that this means specifying any
dependent copies within a batch will result in undefined behavior.

Performs memory copies as specified in the opList array. The length of this array is specified in
numOps. Each entry in this array describes a copy operation. This includes among other things, the
source and destination operands for the copy as specified in CUDA_MEMCPY3D_BATCH_OP::src
and CUDA_MEMCPY3D_BATCH_OP::dst respectively. The source and destination operands of a
copy can either be a pointer or a CUDA array. The width, height and depth of a copy is specified in
CUDA_MEMCPY3D_BATCH_OP::extent. The width, height and depth of a copy are specified in
elements and must not be zero. For pointer-to-pointer copies, the element size is considered to be 1. For
pointer to CUDA array or vice versa copies, the element size is determined by the CUDA array. For
CUDA array to CUDA array copies, the element size of the two CUDA arrays must match.

For a given operand, if CUmemcpy3DOperand::type is specified as
CU_MEMCPY_OPERAND_TYPE_POINTER, then CUmemcpy3DOperand::op::ptr will be used.
The CUmemcpy3DOperand::op::ptr::ptr field must contain the pointer where the copy should
begin. The CUmemcpy3DOperand::op::ptr::rowLength field specifies the length of each row in
elements and must either be zero or be greater than or equal to the width of the copy specified in
CUDA_MEMCPY3D_BATCH_OP::extent::width. The CUmemcpy3DOperand::op::ptr::layerHeight
field specifies the height of each layer and must either be zero or be greater than or equal


CUDA Driver API TRM-06703-001 _vRelease Version  |  217


Modules


to the height of the copy specified in CUDA_MEMCPY3D_BATCH_OP::extent::height.
When either of these values is zero, that aspect of the operand is considered to be tightly
packed according to the copy extent. For managed memory pointers on devices where
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is true or system-allocated
pageable memory on devices where CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
is true, the CUmemcpy3DOperand::op::ptr::locHint field can be used to hint the location of the
operand.

If an operand's type is specified as CU_MEMCPY_OPERAND_TYPE_ARRAY, then
CUmemcpy3DOperand::op::array will be used. The CUmemcpy3DOperand::op::array::array field
specifies the CUDA array and CUmemcpy3DOperand::op::array::offset specifies the 3D offset into
that array where the copy begins.

The CUmemcpyAttributes::srcAccessOrder indicates the source access ordering to be
observed for copies associated with the attribute. If the source access order is set to
CU_MEMCPY_SRC_ACCESS_ORDER_STREAM, then the source will be accessed in stream order.
If the source access order is set to CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL
then it indicates that access to the source pointer can be out of stream order and all accesses must be
complete before the API call returns. This flag is suited for ephemeral sources (ex., stack variables)
when it's known that no prior operations in the stream can be accessing the memory and also that the
lifetime of the memory is limited to the scope that the source variable was declared in. Specifying this
flag allows the driver to optimize the copy and removes the need for the user to synchronize the stream
after the API call. If the source access order is set to CU_MEMCPY_SRC_ACCESS_ORDER_ANY
then it indicates that access to the source pointer can be out of stream order and the accesses can
happen even after the API call returns. This flag is suited for host pointers allocated outside CUDA
(ex., via malloc) when it's known that no prior operations in the stream can be accessing the memory.
Specifying this flag allows the driver to optimize the copy on certain platforms. Each memcopy
operation in opList must have a valid srcAccessOrder setting, otherwise this API will return
CUDA_ERROR_INVALID_VALUE.

The CUmemcpyAttributes::flags field can be used to specify certain flags for copies. Setting the
CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE flag indicates that the associated
copies should preferably overlap with any compute work. Note that this flag is a hint and can be
ignored depending on the platform and other parameters of the copy.











CUDA Driver API TRM-06703-001 _vRelease Version  |  218


Modules