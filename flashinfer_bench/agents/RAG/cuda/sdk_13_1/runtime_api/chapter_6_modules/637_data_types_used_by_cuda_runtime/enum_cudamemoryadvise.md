# enum cudaMemoryAdvise

CUDA Memory Advise values

##### Values

**cudaMemAdviseSetReadMostly = 1**
Data will mostly be read and only occassionally be written to
**cudaMemAdviseUnsetReadMostly = 2**
Undo the effect of cudaMemAdviseSetReadMostly
**cudaMemAdviseSetPreferredLocation = 3**
Set the preferred location for the data as the specified device
**cudaMemAdviseUnsetPreferredLocation = 4**
Clear the preferred location for the data
**cudaMemAdviseSetAccessedBy = 5**
Data will be accessed by the specified device, so prevent page faults as much as possible
**cudaMemAdviseUnsetAccessedBy = 6**
Let the Unified Memory subsystem decide on the page faulting policy for the specified device