# 12.1. Introduction

## 12.1. Introduction[](#introduction-clc "Permalink to this headline")

Compute Capability 10.0 introduces Cluster Launch Control, a new
feature that provides developer with more control over thread block scheduling
by cancelling thread blocks or thread block clusters.

When dealing with problems of variable size, there are two main
approaches to determining the number of kernel thread blocks.

**Approach 1: Fixed Work per Thread Block:**

In this approach, the number of thread blocks is determined by the problem
size, while the amount of work done by each thread block remains constant
or is limited.

Key advantages of this approach:

* Load balancing between SMs.

  In particular, when thread block run-times exhibit variability
  and/or when the number of thread blocks is much larger than what
  the GPU can execute simultaneously (resulting in a low-tail effect),
  this approach allows the GPU scheduler to run more thread blocks
  on some SMs than others.
* Preemption.

  The GPU scheduler can start executing a
  [higher-priority kernel](#stream-priorities),
  even if it is launched after the execution of a lower-priority kernel has
  already begun, by scheduling the higher-priority kernel’s thread blocks
  as the currently running lower-priority kernel’s thread blocks finish.
  It can then return to the lower-priority kernel once the higher-priority
  kernel has finished.

**Approach 2: Fixed Number of Thread Blocks:**

In this approach, often implemented as a block-stride or grid-stride loop,
the number of thread blocks does not directly depend
on the problem size. Instead, the amount of work done by each thread block
is a function of the problem size. Typically, the number of thread blocks is
based on the number of SMs on the GPU where the kernel is executed
and the desired occupancy.

Key advantage of this approach:

* Reduced thread block overheads.

  This approach not only reduces amortized thread block launch latency
  but also minimizes the computational overhead associated with shared
  operations across all thread blocks.
  These overheads can be significantly higher than launch latency overheads.

  For example, in convolution kernels, a prologue for calculating
  convolution coefficients – independent of the thread block index –
  can be computed fewer times due to the fixed number of thread blocks,
  thus reducing redundant computations.

**Cluster Launch Control Approach:**

Cluster Launch Control allows a kernel to request (**cancel**)
the thread block index of a block that has not yet started execution.

This mechanism enables work-stealing among thread blocks:
a thread block attempts to cancel the launch of another thread block
that has not started running yet. If cancellation succeeds,
it “steals” the other thread block’s work by using cancelled block index
to perform the task.

The cancellation will fail if there are no more thread block indices
available and may fail for other reasons, such as a higher-priority kernel
being scheduled. In the latter case, if a thread block exits after a
cancellation failure, the scheduler can start executing the higher-priority
kernel, after which it will continue scheduling the remaining thread blocks
of the current kernel for execution.

The table below summarizes advantages and disadvantages of the three approaches:

|  | **Fixed Work per Thread Block** | **Fixed Number of Thread Blocks** | **Cluster Launch Control** |
| --- | --- | --- | --- |
| Reduced overheads | **\(\textcolor{red}{\textbf{X}}\)** | **\(\textcolor{lime}{\textbf{V}}\)** | **\(\textcolor{lime}{\textbf{V}}\)** |
| Preemption | **\(\textcolor{lime}{\textbf{V}}\)** | **\(\textcolor{red}{\textbf{X}}\)** | **\(\textcolor{lime}{\textbf{V}}\)** |
| Load balancing | **\(\textcolor{lime}{\textbf{V}}\)** | **\(\textcolor{red}{\textbf{X}}\)** | **\(\textcolor{lime}{\textbf{V}}\)** |