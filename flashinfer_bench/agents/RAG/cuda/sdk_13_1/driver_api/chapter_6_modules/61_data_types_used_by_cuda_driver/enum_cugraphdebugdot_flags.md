# enum CUgraphDebugDot_flags

The additional write options for cuGraphDebugDotPrint

###### Values

**CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1<<0**
Output all debug data as if every debug flag is enabled
**CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 1<<1**
Use CUDA Runtime structures for output
**CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 1<<2**
Adds CUDA_KERNEL_NODE_PARAMS values to output
**CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 1<<3**
Adds CUDA_MEMCPY3D values to output
**CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 1<<4**
Adds CUDA_MEMSET_NODE_PARAMS values to output
**CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 1<<5**
Adds CUDA_HOST_NODE_PARAMS values to output
**CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 1<<6**
Adds CUevent handle from record and wait nodes to output
**CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 1<<7**
Adds CUDA_EXT_SEM_SIGNAL_NODE_PARAMS values to output
**CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 1<<8**
Adds CUDA_EXT_SEM_WAIT_NODE_PARAMS values to output
**CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 1<<9**
Adds CUkernelNodeAttrValue values to output
**CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1<<10**
Adds node handles and every kernel function handle to output
**CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 1<<11**
Adds memory alloc node parameters to output
**CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 1<<12**
Adds memory free node parameters to output
**CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 1<<13**


CUDA Driver API TRM-06703-001 _vRelease Version  |  47


Modules


Adds batch mem op node parameters to output
**CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO = 1<<14**
Adds edge numbering information
**CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS = 1<<15**
Adds conditional node parameters to output