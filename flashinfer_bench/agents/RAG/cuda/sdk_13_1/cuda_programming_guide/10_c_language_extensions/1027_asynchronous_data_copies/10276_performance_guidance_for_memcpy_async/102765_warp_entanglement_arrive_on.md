# 10.27.6.5. Warp Entanglement - Arrive-On

#### 10.27.6.5. Warp Entanglement - Arrive-On[](#warp-entanglement-arrive-on "Permalink to this headline")

Warp-divergence affects the number of times an `arrive_on(bar)` operation updates the barrier. If the invoking warp is fully converged, then the barrier is updated once. If the invoking warp is fully diverged, then 32 individual updates are applied to the barrier.