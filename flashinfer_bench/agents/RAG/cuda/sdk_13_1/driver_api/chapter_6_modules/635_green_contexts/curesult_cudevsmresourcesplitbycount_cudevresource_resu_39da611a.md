# CUresult cuDevSmResourceSplitByCount (CUdevResource *result, unsigned int *nbGroups, const CUdevResource *input, CUdevResource *remainder, unsigned int flags, unsigned int minCount)

Splits CU_DEV_RESOURCE_TYPE_SM resources.

###### Parameters

**result**

  - Output array of CUdevResource resources. Can be NULL to query the number of groups.
**nbGroups**

  - This is a pointer, specifying the number of groups that would be or should be created as described
below.
**input**

  - Input SM resource to be split. Must be a valid CU_DEV_RESOURCE_TYPE_SM resource.
**remainder**

  - If the input resource cannot be cleanly split among nbGroups, the remainder is placed in here.
Can be ommitted (NULL) if the user does not need the remaining set.
**flags**

  - Flags specifying how these partitions are used or which constraints to abide by when splitting the
input. Zero is valid for default behavior.
**minCount**

  - Minimum number of SMs required

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_RESOURCE_TYPE,
CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION

###### Description

Splits CU_DEV_RESOURCE_TYPE_SM resources into nbGroups, adhering to the minimum SM
count specified in minCount and the usage flags in flags. If result is NULL, the API simulates
a split and provides the amount of groups that would be created in nbGroups. Otherwise, nbGroups
must point to the amount of elements in result and on return, the API will overwrite nbGroups
with the amount actually created. The groups are written to the array in result. nbGroups can be
less than the total amount if a smaller number of groups is needed.

This API is used to spatially partition the input resource. The input resource needs to come from one of
cuDeviceGetDevResource, cuCtxGetDevResource, or cuGreenCtxGetDevResource. A limitation of the


CUDA Driver API TRM-06703-001 _vRelease Version  |  581


Modules


API is that the output results cannot be split again without first creating a descriptor and a green context
with that descriptor.

When creating the groups, the API will take into account the performance and functional
characteristics of the input resource, and guarantee a split that will create a disjoint set of
symmetrical partitions. This may lead to fewer groups created than purely dividing the
total SM count by the minCount due to cluster requirements or alignment and granularity
requirements for the minCount. These requirements can be queried with cuDeviceGetDevResource,
cuCtxGetDevResource, and cuGreenCtxGetDevResource for CU_DEV_RESOURCE_TYPE_SM,
using the minSmPartitionSize and smCoscheduledAlignment fields to determine
minimum partition size and alignment granularity, respectively.

The remainder set does not have the same functional or performance guarantees as the groups
in result. Its use should be carefully planned and future partitions of the remainder set are
discouraged.

The following flags are supported:

: Lower the minimum SM

###### ‣ CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING

count and alignment, and treat each SM independent of its hierarchy. This allows more fine grained
partitions but at the cost of advanced features (such as large clusters on compute capability 9.0+).
: Compute Capability

###### ‣ CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE

9.0+ only. Attempt to create groups that may allow for maximally sized thread clusters. This can be
queried post green context creation using cuOccupancyMaxPotentialClusterSize.

A successful API call must either have:

A valid array of pointers of size passed in, with of type

###### ‣ result nbGroups input

CU_DEV_RESOURCE_TYPE_SM. Value of minCount must be between 0 and the SM count
specified in input. remainder may be NULL.
NULL passed in for, with a valid integer pointer in and of type

###### ‣ result nbGroups input

CU_DEV_RESOURCE_TYPE_SM. Value of minCount must be between 0 and the SM count
specified in input. remainder may be NULL. This queries the number of groups that would be
created by the API.

Note: The API is not supported on 32-bit platforms.


See also:

cuGreenCtxGetDevResource, cuCtxGetDevResource, cuDeviceGetDevResource


CUDA Driver API TRM-06703-001 _vRelease Version  |  582


Modules