# CUresult cuGraphAddKernelNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams)

Creates a kernel execution node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**nodeParams**

  - Parameters for the GPU execution node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new kernel execution node and adds it to hGraph with numDependencies dependencies
specified via dependencies and arguments specified in nodeParams. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
dependencies may not have any duplicate entries. A handle to the new node will be returned in
phGraphNode.


CUDA Driver API TRM-06703-001 _vRelease Version  |  426


Modules


The CUDA_KERNEL_NODE_PARAMS structure is defined as:


When the graph is launched, the node will invoke kernel func on a (gridDimX x gridDimY x
gridDimZ) grid of blocks. Each block contains (blockDimX x blockDimY x blockDimZ)
threads.

sharedMemBytes sets the amount of dynamic shared memory that will be available to each thread
block.

Kernel parameters to func can be specified in one of two ways:

1) Kernel parameters can be specified via kernelParams. If the kernel has N parameters, then
kernelParams needs to be an array of N pointers. Each pointer, from kernelParams[0] to
kernelParams[N-1], points to the region of memory from which the actual parameter will be
copied. The number of kernel parameters and their offsets and sizes do not need to be specified as that
information is retrieved directly from the kernel's image.

2) Kernel parameters for non-cooperative kernels can also be packaged by the application into a single
buffer that is passed in via extra. This places the burden on the application of knowing each kernel
parameter's size and alignment/padding within the buffer. The extra parameter exists to allow this
function to take additional less commonly used arguments. extra specifies a list of names of extra
settings and their corresponding values. Each extra setting name is immediately followed by the
corresponding value. The list must be terminated with either NULL or CU_LAUNCH_PARAM_END.

CU_LAUNCH_PARAM_END, which indicates the end of the array;

###### ‣ extra

CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next value in will

###### ‣ extra

be a pointer to a buffer containing all the kernel parameters for launching kernel func;
CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next value in

###### **‣**

extra will be a pointer to a size_t containing the size of the buffer specified with
CU_LAUNCH_PARAM_BUFFER_POINTER;

The error CUDA_ERROR_INVALID_VALUE will be returned if kernel parameters are specified
with both kernelParams and extra (i.e. both kernelParams and extra are non-NULL).
CUDA_ERROR_INVALID_VALUE will be returned if extra is used for a cooperative kernel.


CUDA Driver API TRM-06703-001 _vRelease Version  |  427


Modules


The kernelParams or extra array, as well as the argument values it points to, are copied during
this call.


Note:


Kernels launched using graphs must not use texture and surface references. Reading or writing through
any texture or surface reference is undefined behavior. This restriction does not apply to texture and
surface objects.





See also:

cuGraphAddNode, cuLaunchKernel, cuLaunchCooperativeKernel, cuGraphKernelNodeGetParams,
cuGraphKernelNodeSetParams, cuGraphCreate, cuGraphDestroyNode, cuGraphAddChildGraphNode,
cuGraphAddEmptyNode, cuGraphAddHostNode, cuGraphAddMemcpyNode,
cuGraphAddMemsetNode