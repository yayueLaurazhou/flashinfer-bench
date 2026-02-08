# enum CUmemcpySrcAccessOrder

These flags allow applications to convey the source access ordering CUDA must maintain. The
destination will always be accessed in stream order.

###### Values

**CU_MEMCPY_SRC_ACCESS_ORDER_INVALID = 0x0**
Default invalid.
**CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 0x1**


CUDA Driver API TRM-06703-001 _vRelease Version  |  65


Modules


Indicates that access to the source pointer must be in stream order.
**CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL = 0x2**
Indicates that access to the source pointer can be out of stream order and all accesses must be
complete before the API call returns. This flag is suited for ephemeral sources (ex., stack variables)
when it's known that no prior operations in the stream can be accessing the memory and also
that the lifetime of the memory is limited to the scope that the source variable was declared in.
Specifying this flag allows the driver to optimize the copy and removes the need for the user to
synchronize the stream after the API call.
**CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 0x3**
Indicates that access to the source pointer can be out of stream order and the accesses can happen
even after the API call returns. This flag is suited for host pointers allocated outside CUDA (ex.,
via malloc) when it's known that no prior operations in the stream can be accessing the memory.
Specifying this flag allows the driver to optimize the copy on certain platforms.
**CU_MEMCPY_SRC_ACCESS_ORDER_MAX = 0x7FFFFFFF**