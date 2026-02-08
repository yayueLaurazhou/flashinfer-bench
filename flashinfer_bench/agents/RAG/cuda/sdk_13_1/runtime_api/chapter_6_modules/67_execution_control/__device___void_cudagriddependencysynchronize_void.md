# __device__ void cudaGridDependencySynchronize (void)

Programmatic grid dependency synchronization.

##### Description

This device function will block the thread until all direct grid dependencies have completed.
This API is intended to use in conjuncture with programmatic / launch event / dependency.
See cudaLaunchAttributeID::cudaLaunchAttributeProgrammaticStreamSerialization and
cudaLaunchAttributeID::cudaLaunchAttributeProgrammaticEvent for more information.