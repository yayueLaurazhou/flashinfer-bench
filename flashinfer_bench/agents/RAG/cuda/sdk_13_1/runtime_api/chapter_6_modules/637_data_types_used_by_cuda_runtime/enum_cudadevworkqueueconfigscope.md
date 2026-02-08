# enum cudaDevWorkqueueConfigScope

Sharing scope for workqueues

##### Values

**cudaDevWorkqueueConfigScopeDeviceCtx = 0**
Use all shared workqueue resources on the device. Default driver behaviour.
**cudaDevWorkqueueConfigScopeGreenCtxBalanced = 1**
When possible, use non-overlapping workqueue resources with other balanced green contexts.