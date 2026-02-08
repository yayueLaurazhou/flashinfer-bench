# 12.2.1. Thread block cancellation steps

### 12.2.1. Thread block cancellation steps[](#thread-block-cancellation-steps "Permalink to this headline")

The preferred way to use Cluster Launch Control is from a single thread,
i.e., one request at a time.

The following are the five steps of the thread block cancellation process.
The first two steps are declarations and initialization of cancellation result
and synchronization variables, which are done before the work-stealing.
The last three steps are typically executed inside a work-stealing loop
over thread block indices.

1. Declare variables for thread block cancellation:

   ```
   __shared__ uint4 result; // Request result.
   __shared__ uint64_t bar; // Synchronization barrier.
   int phase = 0;           // Synchronization barrier phase.
   ```
2. Initialize shared memory barrier with a single arrival count:

   ```
   if (cg::thread_block::thread_rank() == 0)
       ptx::mbarrier_init(&bar, 1);
   __syncthreads();
   ```
3. Submit asynchronous cancellation request by a single thread and
   set transaction count:

   ```
   if (cg::thread_block::thread_rank() == 0) {
       cg::invoke_one(cg::coalesced_threads(), ptx::clusterlaunchcontrol_try_cancel, &result, &bar);
       ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
   }
   ```

   Note

   Since thread block cancellation is a uniform instruction,
   it is recommended to submit it inside
   [invoke\_one](#invoke-one-and-invoke-one-broadcast) thread selector.
   This allows the compliler to optimize out the peeling loop.
4. Synchronize (complete) asynchronous cancellation request:

   ```
   while (!ptx::mbarrier_try_wait_parity(&bar, phase))
   {}
   phase ^= 1;
   ```
5. Retrieve cancellation status and cancelled thread block index:

   ```
   bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
   if (success) {
       // Don't need all three for 1D/2D thread blocks:
       int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(result);
       int by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y(result);
       int bz = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z(result);
   }
   ```
6. Ensure visibility of shared memory operations between async and generic
   [proxies](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies),
   and protect against data races between iterations of the work-stealing loop.