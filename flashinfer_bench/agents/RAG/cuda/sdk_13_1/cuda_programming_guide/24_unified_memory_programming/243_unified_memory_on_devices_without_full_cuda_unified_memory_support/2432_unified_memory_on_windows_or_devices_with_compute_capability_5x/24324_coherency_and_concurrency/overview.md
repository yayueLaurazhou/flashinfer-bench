# 24.3.2.4. Coherency and Concurrency

#### 24.3.2.4. Coherency and Concurrency[](#coherency-and-concurrency "Permalink to this headline")

Simultaneous access to managed memory on devices of compute capability lower than 6.0 is not possible, because coherence could not be guaranteed if the CPU accessed a Unified Memory allocation while a GPU kernel was active.