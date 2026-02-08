# 10.27.6.3. Warp Entanglement - Commit

#### 10.27.6.3. Warp Entanglement - Commit[](#warp-entanglement-commit "Permalink to this headline")

The sequence of `memcpy_async` batches is shared across the warp. The commit operation is coalesced such that the sequence is incremented once for all converged threads that invoke the commit operation. If the warp is fully converged, the sequence is incremented by one; if the warp is fully diverged, the sequence is incremented by 32.

* Let *PB* be the warp-shared pipeline’s *actual* sequence of batches.

  `PB = {BP0, BP1, BP2, …, BPL}`
* Let *TB* be a thread’s *perceived* sequence of batches, as if the sequence were only incremented by this thread’s invocation of the commit operation.

  `TB = {BT0, BT1, BT2, …, BTL}`

  The `pipeline::producer_commit()` return value is from the thread’s *perceived* batch sequence.
* An index in a thread’s perceived sequence always aligns to an equal or larger index in the actual warp-shared sequence. The sequences are equal only when all commit operations are invoked from converged threads.

  `BTn ≡ BPm` where `n <= m`

For example, when a warp is fully diverged:

* The warp-shared pipeline’s actual sequence would be: `PB = {0, 1, 2, 3, ..., 31}` (`PL=31`).
* The perceived sequence for each thread of this warp would be:

  + Thread 0: `TB = {0}` (`TL=0`)
  + Thread 1: `TB = {0}` (`TL=0`)
  + `…`
  + Thread 31: `TB = {0}` (`TL=0`)