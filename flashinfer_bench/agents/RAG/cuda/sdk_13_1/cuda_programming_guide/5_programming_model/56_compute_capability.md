# 5.6. Compute Capability

## 5.6. Compute Capability[](#compute-capability "Permalink to this headline")

The *compute capability* of a device is represented by a version number, also sometimes called its “SM version”. This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.

The compute capability comprises a major revision number *X* and a minor revision number *Y* and is denoted by *X.Y*.

The major revision number indicates the core GPU architecture of a device. Devices with the same major revision number share the same fundamental architecture. The table below lists the major revision numbers corresponding to each NVIDIA GPU architecture.

Table 2 GPU Architecture and Major Revision Numbers[](#id453 "Permalink to this table")

| Major Revision Number | NVIDIA GPU Architecture |
| --- | --- |
| 9 | NVIDIA Hopper GPU Architecture |
| 8 | NVIDIA Ampere GPU Architecture |
| 7 | NVIDIA Volta GPU Architecture |
| 6 | NVIDIA Pascal GPU Architecture |
| 5 | NVIDIA Maxwell GPU Architecture |
| 3 | NVIDIA Kepler GPU Architecture |

The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new features.

Table 3 Incremental Updates in GPU Architectures[](#id454 "Permalink to this table")

| Compute Capability | NVIDIA GPU Architecture | Based On |
| --- | --- | --- |
| 7.5 | NVIDIA Turing GPU Architecture | NVIDIA Volta GPU Architecture |

[CUDA-Enabled GPUs](#cuda-enabled-gpus) lists of all CUDA-enabled devices along with their compute capability. [Compute Capabilities](#compute-capabilities) gives the technical specifications of each compute capability.

Note

The compute capability version of a particular GPU should not be confused with the CUDA version (for example, CUDA 7.5, CUDA 8, CUDA 9), which is the version of the CUDA *software platform*. The CUDA platform is used by application developers to create applications that run on many generations of GPU architectures, including future GPU architectures yet to be invented. While new versions of the CUDA platform often add native support for a new GPU architecture by supporting the compute capability version of that architecture, new versions of the CUDA platform typically also include software features that are independent of hardware generation.

The *Tesla* and *Fermi* architectures are no longer supported starting with CUDA 7.0 and CUDA 9.0, respectively.