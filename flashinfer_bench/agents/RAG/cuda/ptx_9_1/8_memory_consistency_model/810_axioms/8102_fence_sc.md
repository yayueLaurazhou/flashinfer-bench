# 8.10.2. Fence-SC

### 8.10.2. [Fence-SC](https://docs.nvidia.com/cuda/parallel-thread-execution/#fence-sc-axiom)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#fence-sc-axiom "Permalink to this headline")

*Fence-SC* order cannot contradict *causality order*. For a pair of *morally strong* *fence.sc*
operations F1 and F2, if F1 precedes F2 in *causality order*, then F1 must precede F2 in *Fence-SC*
order.