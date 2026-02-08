# 26.1.1. EGM Platforms: System topology

### 26.1.1. EGM Platforms: System topology[](#egm-platforms-system-topology "Permalink to this headline")

Currently, EGM can be enabled in three platforms: **(1) Single-Node, Single-GPU**:
Consists of an Arm-based CPU, CPU attached memory, and a GPU. Between the CPU
and the GPU there is a high bandwidth C2C (Chip-to-Chip) interconnect.
**(2) Single-Node, Multi-GPU**: Consists of fully connected four
single-node, single-GPU platforms. **(3) Multi-Node, Single-GPU**:
Two or more single-node multi-socket systems.

Note

Using `cgroups` to limit available devices will block routing over EGM
and cause performance issues. Use `CUDA_VISIBLE_DEVICES` instead.