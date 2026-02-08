# enum cudaFuncAttribute

CUDA function attributes that can be set using cudaFuncSetAttribute

##### Values

**cudaFuncAttributeMaxDynamicSharedMemorySize = 8**
Maximum dynamic shared memory size
**cudaFuncAttributePreferredSharedMemoryCarveout = 9**
Preferred shared memory-L1 cache split
**cudaFuncAttributeClusterDimMustBeSet = 10**
Indicator to enforce valid cluster dimension specification on kernel launch
**cudaFuncAttributeRequiredClusterWidth = 11**
Required cluster width
**cudaFuncAttributeRequiredClusterHeight = 12**
Required cluster height
**cudaFuncAttributeRequiredClusterDepth = 13**
Required cluster depth
**cudaFuncAttributeNonPortableClusterSizeAllowed = 14**
Whether non-portable cluster scheduling policy is supported
**cudaFuncAttributeClusterSchedulingPolicyPreference = 15**
Required cluster scheduling policy preference
**cudaFuncAttributeMax**