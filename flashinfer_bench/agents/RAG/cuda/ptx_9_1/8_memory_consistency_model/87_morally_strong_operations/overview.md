# 8.7. Morally strong operations

## 8.7. [Morally strong operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#morally-strong-operations)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#morally-strong-operations "Permalink to this headline")

Two operations are said to be *morally strong* relative to each other if they satisfy all of the
following conditions:

1. The operations are related in *program order* (i.e, they are both executed by the same thread),
   or each operation is *strong* and specifies a *scope* that includes the thread executing the
   other operation.
2. Both operations are performed via the same *proxy*.
3. If both are memory operations, then they overlap completely.

Most (but not all) of the axioms in the memory consistency model depend on relations between
*morally strong* operations.