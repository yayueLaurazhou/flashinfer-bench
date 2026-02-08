# 9.7.13.15. Parallel Synchronization and Communication Instructions: mbarrier

#### 9.7.13.15. [Parallel Synchronization and Communication Instructions: `mbarrier`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier "Permalink to this headline")

`mbarrier` is a barrier created in shared memory that supports :

* Synchronizing any subset of threads within a CTA
* One-way synchronization of threads across CTAs of a cluster. As noted in
  [mbarrier support with shared memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-smem), threads can
  perform only *arrive* operations but not *\*\_wait* on an mbarrier located in `shared::cluster`
  space.
* Waiting for completion of asynchronous memory operations initiated by a thread and making them
  visible to other threads.

An *mbarrier object* is an opaque object in memory which can be initialized and invalidated using :

* `mbarrier.init`
* `mbarrier.inval`

Operations supported on *mbarrier object*s are :

* `mbarrier.expect_tx`
* `mbarrier.complete_tx`
* `mbarrier.arrive`
* `mbarrier.arrive_drop`
* `mbarrier.test_wait`
* `mbarrier.try_wait`
* `mbarrier.pending_count`
* `cp.async.mbarrier.arrive`

Performing any *mbarrier* operation except `mbarrier.init` on an uninitialized *mbarrier object*
results in undefined behavior.
Performing any *non-mbarrier* or `mbarrier.init` operations on an initialized *mbarrier object*
results in undefined behavior.

Unlike `bar{.cta}`/`barrier{.cta}` instructions which can access a limited number of barriers
per CTA, *mbarrier objects* are user defined and are only limited by the total shared memory size
available.

*mbarrier* operations enable threads to perform useful work after the arrival at the *mbarrier* and
before waiting for the *mbarrier* to complete.