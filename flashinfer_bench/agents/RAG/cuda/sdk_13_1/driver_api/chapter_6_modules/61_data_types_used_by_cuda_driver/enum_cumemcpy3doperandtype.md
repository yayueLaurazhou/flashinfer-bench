# enum CUmemcpy3DOperandType

These flags allow applications to convey the operand type for individual copies specified in
cuMemcpy3DBatchAsync.

###### Values

**CU_MEMCPY_OPERAND_TYPE_POINTER = 0x1**
Memcpy operand is a valid pointer.
**CU_MEMCPY_OPERAND_TYPE_ARRAY = 0x2**
Memcpy operand is a CUarray.
**CU_MEMCPY_OPERAND_TYPE_MAX = 0x7FFFFFFF**