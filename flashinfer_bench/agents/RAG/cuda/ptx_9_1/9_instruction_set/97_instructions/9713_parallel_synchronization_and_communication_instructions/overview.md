# 9.7.13. Parallel Synchronization and Communication Instructions

### 9.7.13. [Parallel Synchronization and Communication Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions "Permalink to this headline")

These instructions are:

* `bar{.cta}`, `barrier{.cta}`
* `bar.warp.sync`
* `barrier.cluster`
* `membar`
* `atom`
* `red`
* `red.async`
* `vote`
* `match.sync`
* `activemask`
* `redux.sync`
* `griddepcontrol`
* `elect.sync`
* `mbarrier.init`
* `mbarrier.inval`
* `mbarrier.arrive`
* `mbarrier.arrive_drop`
* `mbarrier.test_wait`
* `mbarrier.try_wait`
* `mbarrier.pending_count`
* `cp.async.mbarrier.arrive`
* `tensormap.cp_fenceproxy`
* `clusterlaunchcontrol.try_cancel`
* `clusterlaunchcontrol.query_cancel`