# enum cudaMemcpy3DOperandType

These flags allow applications to convey the operand type for individual copies specified in
cudaMemcpy3DBatchAsync.

##### Values

**cudaMemcpyOperandTypePointer = 0x1**
Memcpy operand is a valid pointer.
**cudaMemcpyOperandTypeArray = 0x2**
Memcpy operand is a CUarray.
**cudaMemcpyOperandTypeMax = 0x7FFFFFFF**