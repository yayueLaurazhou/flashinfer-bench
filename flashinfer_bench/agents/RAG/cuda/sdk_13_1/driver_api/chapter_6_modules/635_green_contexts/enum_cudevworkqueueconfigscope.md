# enum CUdevWorkqueueConfigScope

Sharing scope for workqueues

###### Values

**CU_WORKQUEUE_SCOPE_DEVICE_CTX = 0**
Use all shared workqueue resources across all contexts. Default driver behaviour.
**CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED = 1**
When possible, use non-overlapping workqueue resources with other balanced green contexts.


CUDA Driver API TRM-06703-001 _vRelease Version  |  574


Modules