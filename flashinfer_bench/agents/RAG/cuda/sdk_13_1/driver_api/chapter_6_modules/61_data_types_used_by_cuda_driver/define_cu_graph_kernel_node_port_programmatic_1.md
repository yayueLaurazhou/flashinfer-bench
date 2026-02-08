# #define CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC 1

This port activates when all blocks of the kernel have performed
cudaTriggerProgrammaticLaunchCompletion() or have terminated. It must be used
with edge type CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC. See also
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT.


CUDA Driver API TRM-06703-001 _vRelease Version  |  90


Modules