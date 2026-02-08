# #define CU_MEMHOSTREGISTER_READ_ONLY 0x08

If set, the passed memory pointer is treated as pointing to memory
that is considered read-only by the device. On platforms without
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
this flag is required in order to register memory mapped to the CPU as readonly. Support for the use of this flag can be queried from the device attribute
CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED. Using this
flag with a current context associated with a device that does not have this attribute set will cause
cuMemHostRegister to error with CUDA_ERROR_NOT_SUPPORTED.