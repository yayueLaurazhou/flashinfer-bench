# enum CUstreamAtomicReductionOpType

Atomic reduction operation types for CUstreamBatchMemOpParams::atomicReduction::reductionOp

###### Values

**CU_STREAM_ATOMIC_REDUCTION_OP_OR = CU_ATOMIC_OPERATION_OR**
Performs an atomic OR: *(address) = *(address) | value
**CU_STREAM_ATOMIC_REDUCTION_OP_AND = CU_ATOMIC_OPERATION_AND**
Performs an atomic AND: *(address) = *(address) & value
**CU_STREAM_ATOMIC_REDUCTION_OP_ADD =**
**CU_ATOMIC_OPERATION_INTEGER_ADD**
Performs an atomic ADD: *(address) = *(address) + value