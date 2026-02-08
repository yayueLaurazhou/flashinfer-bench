# __device__ void cudaTriggerProgrammaticLaunchCompletion (void)

Programmatic dependency trigger.

##### Description

This device function ensures the programmatic launch completion edges / events are fulfilled.
See cudaLaunchAttributeID::cudaLaunchAttributeProgrammaticStreamSerialization and
cudaLaunchAttributeID::cudaLaunchAttributeProgrammaticEvent for more information. The event /
edge kick off only happens when every CTAs in the grid has either exited or called this function at
least once, otherwise the kick off happens automatically after all warps finishes execution but before
the grid completes. The kick off only enables scheduling of the secondary kernel. It provides no
memory visibility guarantee itself. The user could enforce memory visibility by inserting a memory
fence of the correct scope.


CUDA Runtime API vRelease Version  |  109


Modules