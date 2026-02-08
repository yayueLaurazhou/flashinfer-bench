# ‣ cudaDevWorkqueueConfigScopeDeviceCtx:

all contexts (default driver behavior).
When possible, use non##### ‣ cudaDevWorkqueueConfigScopeGreenCtxBalanced:
overlapping workqueue resources with other balanced green contexts.

The maximum concurrency limit depends on CUDA_DEVICE_MAX_CONNECTIONS and can be
queried from the device via cudaDeviceGetDevResource. Configurations may exceed this concurrency
limit, but the driver will not guarantee that work submission remains non-overlapping.

For cudaDevResourceTypeWorkqueue, the resource represents a pre-existing workqueue that
can be retrieved from existing execution contexts. This allows reusing workqueue resources across
different execution contexts.

On Concurrency

Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels
launched in them will run concurrently or have forward progress guarantees. This is due to
other resources that could cause a dependency. Using a combination of disjoint SMs and
cudaDevWorkqueueConfigScopeGreenCtxBalanced workqueue configurations can provide
the best chance of avoiding interference. More resources will be added in the future to provide stronger
guarantees.

Additionally, there are two known scenarios, where its possible for the workload to run on more SMs
than was provisioned (but never less).

On Volta+ MPS: When is used, the set of SMs