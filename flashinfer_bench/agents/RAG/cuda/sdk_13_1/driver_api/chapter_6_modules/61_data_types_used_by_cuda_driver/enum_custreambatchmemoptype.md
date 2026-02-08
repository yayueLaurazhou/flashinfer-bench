# enum CUstreamBatchMemOpType

Operations for cuStreamBatchMemOp

###### Values

**CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1**
Represents a cuStreamWaitValue32 operation
**CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2**
Represents a cuStreamWriteValue32 operation
**CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4**
Represents a cuStreamWaitValue64 operation
**CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5**
Represents a cuStreamWriteValue64 operation
**CU_STREAM_MEM_OP_BARRIER = 6**
Insert a memory barrier of the specified type


CUDA Driver API TRM-06703-001 _vRelease Version  |  81


Modules


**CU_STREAM_MEM_OP_ATOMIC_REDUCTION = 8**
Perform a atomic reduction. See CUstreamBatchMemOpParams::atomicReduction
**CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3**
This has the same effect as CU_STREAM_WAIT_VALUE_FLUSH, but as a standalone operation.