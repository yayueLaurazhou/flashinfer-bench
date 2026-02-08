# int cudaPointerAttributes::device

The device against which the memory was allocated or registered. If the memory type is
cudaMemoryTypeDevice then this identifies the device on which the memory referred physically


CUDA Runtime API vRelease Version  |  642


Data Structures


resides. If the memory type is cudaMemoryTypeHost or::cudaMemoryTypeManaged then this
identifies the device which was current when the memory was allocated or registered (and if that device
is deinitialized then this allocation will vanish with that device's state).