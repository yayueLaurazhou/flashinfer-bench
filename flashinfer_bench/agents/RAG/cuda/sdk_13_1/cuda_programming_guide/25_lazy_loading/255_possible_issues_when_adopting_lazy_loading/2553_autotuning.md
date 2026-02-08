# 25.5.3. Autotuning

### 25.5.3. Autotuning[](#autotuning "Permalink to this headline")

Some applications launch several kernels implementing the same functionality to determine which one is the fastest.
While it is overall advisable to run at least one warmup iteration, it becomes especially important with Lazy Loading.
After all, including time taken to load the kernel will skew your results.

Possible solutions:

* do at least one warmup interaction prior to measurement
* preload the benchmarked kernel prior to launching it