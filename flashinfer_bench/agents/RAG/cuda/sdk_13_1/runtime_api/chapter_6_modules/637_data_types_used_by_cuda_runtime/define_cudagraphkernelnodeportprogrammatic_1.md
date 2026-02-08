# #define cudaGraphKernelNodePortProgrammatic 1

This port activates when all blocks of the kernel have performed
cudaTriggerProgrammaticLaunchCompletion() or have terminated. It must be used with edge type
cudaGraphDependencyTypeProgrammatic. See also cudaLaunchAttributeProgrammaticEvent.