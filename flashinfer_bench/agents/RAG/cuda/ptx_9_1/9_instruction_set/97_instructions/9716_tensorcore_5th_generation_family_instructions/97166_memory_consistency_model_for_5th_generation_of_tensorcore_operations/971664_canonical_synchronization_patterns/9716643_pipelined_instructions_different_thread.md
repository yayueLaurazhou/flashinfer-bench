# 9.7.16.6.4.3. Pipelined instructions, different thread

###### 9.7.16.6.4.3. [Pipelined instructions, different thread](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-diff-thread)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-diff-thread "Permalink to this headline")

In this pattern, no explicit waiting mechanism is needed but proper synchronization between threads is needed.

Example:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.cp tcgen05.fence::before_thread_sync mbarrier.arrive.relaxed.cluster ```  Copy to clipboard |  |
|  | ``` mbarrier.try_wait.relaxed.cluster // loop till success tcgen05.fence::after_thread_sync tcgen05.mma ```  Copy to clipboard |