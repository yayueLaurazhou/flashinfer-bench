# 8.9.1. Program Order

### 8.9.1. [Program Order](https://docs.nvidia.com/cuda/parallel-thread-execution/#program-order)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#program-order "Permalink to this headline")

The *program order* relates all operations performed by a thread to the order in which a sequential
processor will execute instructions in the corresponding PTX source. It is a transitive relation
that forms a total order over the operations performed by the thread, but does not relate operations
from different threads.