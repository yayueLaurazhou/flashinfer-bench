# 6.2.8.6.1. Background

##### 6.2.8.6.1. Background[](#background "Permalink to this headline")

A CUDA application utilizes the GPU by launching and executing multiple kernels on it.
A typical GPU activity timeline is shown in [Figure 10](#gpu-activity).

[![GPU activity timeline](_images/gpu-activity.png)](_images/gpu-activity.png)


Figure 10 GPU activity timeline[](#gpu-activity "Permalink to this image")

Here, `secondary_kernel` is launched after `primary_kernel` finishes its execution.
Serialized execution is usually necessary because `secondary_kernel` depends on result data
produced by `primary_kernel`. If `secondary_kernel` has no dependency on `primary_kernel`,
both of them can be launched concurrently by using [Streams](#streams).
Even if `secondary_kernel` is dependent on `primary_kernel`, there is some potential for
concurrent execution. For example, almost all the kernels have
some sort of *preamble* section during which tasks such as zeroing buffers or loading
constant values are performed.

[![Preamble section of ``secondary_kernel``](_images/secondary-kernel-preamble.png)](_images/secondary-kernel-preamble.png)


Figure 11 Preamble section of `secondary_kernel`[](#secondary-kernel-preamble "Permalink to this image")

[Figure 11](#secondary-kernel-preamble) demonstrates the portion of `secondary_kernel` that could
be executed concurrently without impacting the application.
Note that concurrent launch also allows us to hide the launch latency of `secondary_kernel` behind
the execution of `primary_kernel`.

[![Concurrent execution of ``primary_kernel`` and ``secondary_kernel``](_images/preamble-overlap.png)](_images/preamble-overlap.png)


Figure 12 Concurrent execution of `primary_kernel` and `secondary_kernel`[](#preamble-overlap "Permalink to this image")

The concurrent launch and execution of `secondary_kernel` shown in [Figure 12](#preamble-overlap) is
achievable using *Programmatic Dependent Launch*.

*Programmatic Dependent Launch* introduces changes to the CUDA kernel launch APIs as explained in following section.
These APIs require at least compute capability 9.0 to provide overlapping execution.