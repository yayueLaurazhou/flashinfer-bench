# 6.33. Execution Context Management

This section describes the execution context management functions of the CUDA runtime application
programming interface.

Overview

A CUDA execution context cudaExecutionContext_t serves as an abstraction for the contexts exposed
by the CUDA Runtime, specifically green contexts and the primary context, and provides a unified
programming model and API interface for contexts in the Runtime.

There are two primary ways today to obtain an execution context:

cudaDeviceGetExecutionCtx: Returns the execution context that corresponds to the primary