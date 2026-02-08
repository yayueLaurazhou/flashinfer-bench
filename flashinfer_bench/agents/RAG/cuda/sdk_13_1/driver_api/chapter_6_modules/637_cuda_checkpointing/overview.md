# 6.37. CUDA Checkpointing

CUDA API versioning support

This sections describes the checkpoint and restore functions of the low-level CUDA driver application
programming interface.

The CUDA checkpoint and restore API's provide a way to save and restore GPU state for full process
checkpoints when used with CPU side process checkpointing solutions. They can also be used to pause
GPU work and suspend a CUDA process to allow other applications to make use of GPU resources.

Checkpoint and restore capabilities are currently restricted to Linux.


CUDA Driver API TRM-06703-001 _vRelease Version  |  593


Modules