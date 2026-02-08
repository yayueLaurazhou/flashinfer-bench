# 6.2.9.4.1. IOMMU on Linux

##### 6.2.9.4.1. IOMMU on Linux[](#iommu-on-linux "Permalink to this headline")

On Linux only, CUDA and the display driver does not support IOMMU-enabled bare-metal PCIe peer to peer memory copy. However, CUDA and the display driver does support IOMMU via VM pass through. As a consequence, users on Linux, when running on a native bare metal system, should disable the IOMMU. The IOMMU should be enabled and the VFIO driver be used as a PCIe pass through for virtual machines.

On Windows the above limitation does not exist.

See also [Allocating DMA Buffers on 64-bit Platforms](https://download.nvidia.com/XFree86/Linux-x86_64/396.51/README/dma_issues.html).