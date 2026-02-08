# __host__cudaError_t cudaDevSmResourceSplitByCount (cudaDevResource *result, unsigned int *nbGroups, const cudaDevResource *input, cudaDevResource *remaining, unsigned int flags, unsigned int minCount)

Splits cudaDevResourceTypeSm resources.

##### Parameters

**result**

  - Output array of cudaDevResource resources. Can be NULL to query the number of groups.
**nbGroups**

  - This is a pointer, specifying the number of groups that would be or should be created as described
below.
**input**

  - Input SM resource to be split. Must be a valid cudaDevSmResource resource.
**remaining**

  - If the input resource cannot be cleanly split among nbGroups, the remaining is placed in here.
Can be ommitted (NULL) if the user does not need the remaining set.
**flags**

  - Flags specifying how these partitions are used or which constraints to abide by when splitting the
input. Zero is valid for default behavior.
**minCount**

  - Minimum number of SMs required

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotPermitted, cudaErrorInvalidResourceType,
cudaErrorInvalidResourceConfiguration, cudaErrorNotSupported, cudaErrorCudartUnloading,
cudaErrorInitializationError

##### Description

Splits cudaDevResourceTypeSm resources into nbGroups, adhering to the minimum SM count
specified in minCount and the usage flags in flags. If result is NULL, the API simulates a split
and provides the amount of groups that would be created in nbGroups. Otherwise, nbGroups must
point to the amount of elements in result and on return, the API will overwrite nbGroups with the
amount actually created. The groups are written to the array in result. nbGroups can be less than
the total amount if a smaller number of groups is needed.

This API is used to spatially partition the input resource. The input resource needs to come from one
of cudaDeviceGetDevResource, or cudaExecutionCtxGetDevResource. A limitation of the API is that


CUDA Runtime API vRelease Version  |  450


Modules


the output results cannot be split again without first creating a descriptor and a green context with that
descriptor.

When creating the groups, the API will take into account the performance and functional characteristics
of the input resource, and guarantee a split that will create a disjoint set of symmetrical partitions. This
may lead to fewer groups created than purely dividing the total SM count by the minCount due to
cluster requirements or alignment and granularity requirements for the minCount. These requirements
can be queried with cudaDeviceGetDevResource, or cudaExecutionCtxGetDevResource for
cudaDevResourceTypeSm, using the minSmPartitionSize and smCoscheduledAlignment
fields to determine minimum partition size and alignment granularity, respectively.

The remainder set does not have the same functional or performance guarantees as the groups
in result. Its use should be carefully planned and future partitions of the remainder set are
discouraged.

The following flags are supported:

: Lower the minimum SM count

##### ‣ cudaDevSmResourceSplitIgnoreSmCoscheduling

and alignment, and treat each SM independent of its hierarchy. This allows more fine grained
partitions but at the cost of advanced features (such as large clusters on compute capability 9.0+).
: Compute Capability 9.0+

##### ‣ cudaDevSmResourceSplitMaxPotentialClusterSize

only. Attempt to create groups that may allow for maximally sized thread clusters. This can be
queried post green context creation using cudaOccupancyMaxPotentialClusterSize.

A successful API call must either have:

A valid array of pointers of size passed in, with of type

##### ‣ result nbGroups input

cudaDevResourceTypeSm. Value of minCount must be between 0 and the SM count
specified in input. remaining may be NULL.
NULL passed in for, with a valid integer pointer in and of type

##### ‣ result nbGroups input

cudaDevResourceTypeSm. Value of minCount must be between 0 and the SM count
specified in input. remaining may be NULL. This queries the number of groups that would be
created by the API.

Note: The API is not supported on 32-bit platforms.





See also:

cuDevSmResourceSplitByCount, cudaDeviceGetDevResource, cudaExecutionCtxGetDevResource,
cudaDevResourceGenerateDesc


CUDA Runtime API vRelease Version  |  451


Modules