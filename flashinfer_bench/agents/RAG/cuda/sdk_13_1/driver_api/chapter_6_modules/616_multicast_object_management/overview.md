# 6.16. Multicast Object Management

This section describes the CUDA multicast object operations exposed by the low-level CUDA driver
application programming interface.

overview

A multicast object created via cuMulticastCreate enables certain memory operations to be broadcast
to a team of devices. Devices can be added to a multicast object via cuMulticastAddDevice. Memory
can be bound on each participating device via cuMulticastBindMem, cuMulticastBindMem_v2,
cuMulticastBindAddr, or cuMulticastBindAddr_v2. Multicast objects can be mapped into a
device's virtual address space using the virtual memmory management APIs (see cuMemMap and
cuMemSetAccess).

Supported Platforms

Support for multicast on a specific device can be queried using the device attribute
CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED


CUDA Driver API TRM-06703-001 _vRelease Version  |  301


Modules