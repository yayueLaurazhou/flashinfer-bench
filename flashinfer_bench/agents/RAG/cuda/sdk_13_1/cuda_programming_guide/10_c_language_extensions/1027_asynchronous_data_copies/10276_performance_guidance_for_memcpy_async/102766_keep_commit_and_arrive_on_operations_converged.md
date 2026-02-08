# 10.27.6.6. Keep Commit and Arrive-On Operations Converged

#### 10.27.6.6. Keep Commit and Arrive-On Operations Converged[](#keep-commit-and-arrive-on-operations-converged "Permalink to this headline")

It is recommended that commit and arrive-on invocations are by converged threads:

* to not over-wait, by keeping threads’ perceived sequence of batches aligned with the actual sequence, and
* to minimize updates to the barrier object.

When code preceding these operations diverges threads, then the warp should be re-converged, via `__syncwarp` before invoking commit or arrive-on operations.