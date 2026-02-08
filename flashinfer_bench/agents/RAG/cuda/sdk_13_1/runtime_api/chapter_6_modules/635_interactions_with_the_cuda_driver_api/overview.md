# 6.35. Interactions with the CUDA Driver API

This section describes the interactions between the CUDA Driver API and the CUDA Runtime API

Execution Contexts

The CUDA Runtime provides cudaExecutionContext_t as an abstraction over driver-level contexts—
specifically, green contexts and the primary context.

There are two primary ways to obtain an execution context:

cudaDeviceGetExecutionCtx: Returns the execution context that corresponds to the primary