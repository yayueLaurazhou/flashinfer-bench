# 6.15. Stream Ordered Memory Allocator

This section describes the stream ordered memory allocator exposed by the low-level CUDA driver
application programming interface.

overview

The asynchronous allocator allows the user to allocate and free in stream order. All asynchronous
accesses of the allocation must happen between the stream executions of the allocation and the free.
If the memory is accessed outside of the promised stream order, a use before allocation / use after free
error will cause undefined behavior.

The allocator is free to reallocate the memory as long as it can guarantee that compliant memory
accesses will not overlap temporally. The allocator may refer to internal stream ordering as well as
inter-stream dependencies (such as CUDA events and null stream dependencies) when establishing the
temporal guarantee. The allocator may also insert inter-stream dependencies to establish the temporal
guarantee.

Supported Platforms

Whether or not a device supports the integrated stream ordered memory allocator
may be queried by calling cuDeviceGetAttribute() with the device attribute
CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED


CUDA Driver API TRM-06703-001 _vRelease Version  |  286


Modules