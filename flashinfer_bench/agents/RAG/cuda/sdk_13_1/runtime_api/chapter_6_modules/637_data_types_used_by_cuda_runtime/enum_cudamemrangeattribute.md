# enum cudaMemRangeAttribute

CUDA range attributes

##### Values

**cudaMemRangeAttributeReadMostly = 1**


CUDA Runtime API vRelease Version  |  573


Modules


Whether the range will mostly be read and only occassionally be written to
**cudaMemRangeAttributePreferredLocation = 2**
The preferred location of the range
**cudaMemRangeAttributeAccessedBy = 3**
Memory range has cudaMemAdviseSetAccessedBy set for specified device
**cudaMemRangeAttributeLastPrefetchLocation = 4**
The last location to which the range was prefetched
**cudaMemRangeAttributePreferredLocationType = 5**
The preferred location type of the range
**cudaMemRangeAttributePreferredLocationId = 6**
The preferred location id of the range
**cudaMemRangeAttributeLastPrefetchLocationType = 7**
The last location type to which the range was prefetched
**cudaMemRangeAttributeLastPrefetchLocationId = 8**
The last location id to which the range was prefetched