# 9.7.9.25.1. Completion Mechanisms for Asynchronous Copy Operations

##### 9.7.9.25.1. [Completion Mechanisms for Asynchronous Copy Operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms "Permalink to this headline")

A thread must explicitly wait for the completion of an asynchronous copy operation in order to
access the result of the operation. Once an asynchronous copy operation is initiated, modifying the
source memory location or tensor descriptor or reading from the destination memory location before
the asynchronous operation completes, exhibits undefined behavior.

This section describes two asynchronous copy operation completion mechanisms supported in PTX:
Async-group mechanism and mbarrier-based mechanism.

Asynchronous operations may be tracked by either of the completion mechanisms or both mechanisms.
The tracking mechanism is instruction/instruction-variant specific.