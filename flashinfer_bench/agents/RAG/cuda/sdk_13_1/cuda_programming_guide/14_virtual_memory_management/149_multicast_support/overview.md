# 14.9. Multicast Support

## 14.9. Multicast Support[](#multicast-support "Permalink to this headline")

The [Multicast Object Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST/)
provide a way for the application to create Multicast Objects and in combination with the [Virtual Memory Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html/)
described above allow applications to leverage NVLINK SHARP on supported NVLINK connected GPUs if they are connected with NVSWITCH. NVLINK SHARP
allows CUDA applications to leverage in fabric computing to accelerate operations like broadcast and reductions between GPUs connected with
NVSWITCH. For this to work multiple NVLINK connected GPUs form a Multicast Team and each GPU from the team backs up a Multicast Object with
physical memory. So a Multicast Team of N GPUs has N physical replicas, each local to one participating GPU, of a Multicast Object.
The [multimem PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/)
using mappings of Multicast Objects work with all replicas of the Multicast Object.

To work with Multicast Objects an application needs to

* Query Multicast Support
* Create a Multicast Handle with `cuMulticastCreate`.
* Share the Multicast Handle with all processes that control a GPU which should participate in a Multicast Team. This works with `cuMemExportToShareableHandle` as described above.
* Add all GPUs that should participate in the Multicast Team with `cuMulticastAddDevice`.
* For each participating GPU bind physical memory allocated with `cuMemCreate` as described above to the Multicast Handle. All devices need to be added to the Multicast Team before binding memory on any device.
* Reserve an address range, map the Multicast Handle and set Access Rights as described above for regular Unicast mappings. Unicast and Multicast mappings to the same physical memory are possible. See the [Virtual Aliasing Support](#virtual-aliasing-support) section above how to ensure consistency between multiple mappings to the same physical memory.
* Use the [multimem PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/) with the multicast mappings.

The `multi_node_p2p` example in the [Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models/) GitHub
repository contains a complete example using Fabric Memory including Multicast Objects to leverage NVLINK SHARP. Please note that this example is
for developers of libraries like NCCL or NVSHMEM. It shows how higher-level programming models like NVSHMEM work internally within a (multinode)
NVLINK domain. Application developers generally should use the higher-level MPI, NCCL, or NVSHMEM interfaces instead of this API.