# __host__cudaError_t cudaGraphAddKernelNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const cudaKernelNodeParams *pNodeParams)

Creates a kernel execution node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**pNodeParams**

  - Parameters for the GPU execution node

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDeviceFunction

##### Description

Creates a new kernel execution node and adds it to graph with numDependencies dependencies
specified via pDependencies and arguments specified in pNodeParams. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
pDependencies may not have any duplicate entries. A handle to the new node will be returned in
pGraphNode.

The cudaKernelNodeParams structure is defined as:

‎ struct cudaKernelNodeParams


CUDA Runtime API vRelease Version  |  333


Modules


When the graph is launched, the node will invoke kernel func on a (gridDim.x x gridDim.y x
gridDim.z) grid of blocks. Each block contains (blockDim.x x blockDim.y x blockDim.z)
threads.

sharedMem sets the amount of dynamic shared memory that will be available to each thread block.

Kernel parameters to func can be specified in one of two ways:

1) Kernel parameters can be specified via kernelParams. If the kernel has N parameters, then
kernelParams needs to be an array of N pointers. Each pointer, from kernelParams[0] to
kernelParams[N-1], points to the region of memory from which the actual parameter will be
copied. The number of kernel parameters and their offsets and sizes do not need to be specified as that
information is retrieved directly from the kernel's image.

2) Kernel parameters can also be packaged by the application into a single buffer that is passed in
via extra. This places the burden on the application of knowing each kernel parameter's size and
alignment/padding within the buffer. The extra parameter exists to allow this function to take
additional less commonly used arguments. extra specifies a list of names of extra settings and their
corresponding values. Each extra setting name is immediately followed by the corresponding value.
The list must be terminated with either NULL or CU_LAUNCH_PARAM_END.

CU_LAUNCH_PARAM_END, which indicates the end of the array;

##### ‣ extra

CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next value in will

##### ‣ extra

be a pointer to a buffer containing all the kernel parameters for launching kernel func;
CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next value in

##### **‣**

extra will be a pointer to a size_t containing the size of the buffer specified with
CU_LAUNCH_PARAM_BUFFER_POINTER;

The error cudaErrorInvalidValue will be returned if kernel parameters are specified with both
kernelParams and extra (i.e. both kernelParams and extra are non-NULL).

The kernelParams or extra array, as well as the argument values it points to, are copied during
this call.


Note:


Kernels launched using graphs must not use texture and surface references. Reading or writing through
any texture or surface reference is undefined behavior. This restriction does not apply to texture and
surface objects.


CUDA Runtime API vRelease Version  |  334


Modules













See also:

cudaGraphAddNode, cudaLaunchKernel, cudaGraphKernelNodeGetParams,
cudaGraphKernelNodeSetParams, cudaGraphCreate, cudaGraphDestroyNode,
cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddHostNode,
cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode