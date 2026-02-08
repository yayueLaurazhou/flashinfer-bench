# __host__cudaError_t cudaDevSmResourceSplit (cudaDevResource *result, unsigned int nbGroups, const cudaDevResource *input, cudaDevResource *remainder, unsigned int flags, cudaDevSmResourceGroupParams *groupParams)

Splits a cudaDevResourceTypeSm resource into structured groups.

##### Parameters

**result**

  - Output array of cudaDevResource resources. Can be NULL, alongside an smCount of 0,
for discovery purpose.
**nbGroups**

  - Specifies the number of groups in result and groupParams
**input**

  - Input SM resource to be split. Must be a valid cudaDevResourceTypeSm resource.
**remainder**

  - If splitting the input resource leaves any SMs, the remainder is placed in here.
**flags**

  - Flags specifying how the API should behave. The value should be 0 for now.
**groupParams**

  - Description of how the SMs should be split and assigned to the corresponding result entry.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotPermitted, cudaErrorInvalidResourceType,
cudaErrorInvalidResourceConfiguration, cudaErrorNotSupported, cudaErrorCudartUnloading,
cudaErrorInitializationError

##### Description

This API will split a resource of cudaDevResourceTypeSm into nbGroups structured device
resource groups (the result array), as well as an optional remainder, according to a set of
requirements specified in the groupParams array. The term “structured” is a trait that specifies
the result has SMs that are co-scheduled together. This co-scheduling can be specified via the


CUDA Runtime API vRelease Version  |  447


Modules


coscheduledSmCount field of the groupParams structure, while the smCount will specify
how many SMs are required in total for that result. The remainder is always “unstructured”, it does
not have any set guarantees with respect to co-scheduling and those properties will need to either be
queried via the occupancy set of APIs or further split into structured groups by this API.

The API has a discovery mode for use cases where it is difficult to know ahead of time what the SM
count should be. Discovery happens when the smCount field of a given groupParams array entry is
set to 0 - the smCount will be filled in by the API with the derived SM count according to the provided
groupParams fields and constraints. Discovery can be used with both a valid result array and with
a NULL result pointer value. The latter is useful in situations where the smCount will end up being
zero, which is an invalid value to create a result entry with, but allowed for discovery purposes when
the result is NULL.

The groupParams array is evaluated from index 0 to nbGroups - 1. For each index in the
groupParams array, the API will evaluate which SMs may be a good fit based on constraints and
assign those SMs to result. This evaluation order is important to consider when using discovery
mode, as it helps discover the remaining SMs.

For a valid call:

should point to a array of size, or alternatively,

##### ‣ result cudaDevResource nbGroups

may be NULL, if the developer wishes for only the groupParams entries to be updated

should be a valid cudaDevResourceTypeSm resource that originates from querying the

##### ‣ input

execution context, or device.



The group may be NULL.

##### ‣ remainder

There are no API at this time, so the value passed in should be 0.

##### ‣ flags

A cudaDevSmResourceGroupParams array of size . Each entry must be zero##### ‣ nbGroups
initialized.



must be either 0 or in the range of [2,inputSmCount] where inputSmCount is

##### ‣ smCount:

the amount of SMs the input resource has. smCount must be a multiple of 2, as well as
a multiple of coscheduledSmCount. When assigning SMs to a group (and if results are
expected by having the result parameter set), smCount cannot end up with 0 or a value
less than coscheduledSmCount otherwise cudaErrorInvalidResourceConfiguration will be
returned.
allows grouping SMs together in order to be able to launch

##### ‣ coscheduledSmCount:

clusters on Compute Architecture 9.0+. The default value may be queried from the device’s
cudaDevResourceTypeSm resource (8 on Compute Architecture 9.0+ and 2 otherwise). The
maximum is 32 on Compute Architecture 9.0+ and 2 otherwise.
Attempts to merge

##### ‣ preferredCoscheduledSmCount: coscheduledSmCount

groups into larger groups, in order to make use of preferredClusterDimensions on
Compute Architecture 10.0+. The default value is set to coscheduledSmCount.

##### ‣ flags:

CUDA Runtime API vRelease Version  |  448


Modules


lets be a non-multiple of

##### ‣ cudaDevSmResourceGroupBackfill: smCount

coscheduledSmCount, filling the difference between SM count and already assigned
co-scheduled groupings with other SMs. This lets any resulting group behave similar to the
remainder group for example.

Example params and their effect:

A groupParams array element is defined in the following order:
‎ { .smCount, .coscheduledSmCount, .preferredCoscheduledSmCount, .flags, \/
\* .reserved \*\/ }


The difference between a catch-all param group as the last entry and the remainder is in two aspects:

The remainder may be NULL / _TYPE_INVALID (if there are no SMs remaining), while a result

##### **‣**

group must always be valid.
The remainder does not have a structure, while the result group will always need

##### **‣**

to adhere to a structure of coscheduledSmCount (even if its just 2), and therefore
must always have enough coscheduled SMs to cover that requirement (even with the
cudaDevSmResourceGroupBackfill flag enabled).

Splitting an input into N groups, can be accomplished by repeatedly splitting off 1 group and resplitting the remainder (a bisect operation). However, it's recommended to accomplish this with a
single call wherever possible.





See also:


CUDA Runtime API vRelease Version  |  449


Modules


cuDevSmResourceSplit, cudaDeviceGetDevResource, cudaExecutionCtxGetDevResource,
cudaDevResourceGenerateDesc