# CUlaunchAttributeValue::@9 CUlaunchAttributeValue::preferredClusterDim

Value of launch attribute CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION
that represents the desired preferred cluster dimensions for the kernel. Opaque type with the following
fields:

   - The X dimension of the preferred cluster, in blocks. Must be a divisor of the grid X dimension,

###### ‣ x

and must be a multiple of the x field of CUlaunchAttributeValue::clusterDim.
   - The Y dimension of the preferred cluster, in blocks. Must be a divisor of the grid Y dimension,

###### ‣ y

and must be a multiple of the y field of CUlaunchAttributeValue::clusterDim.
   - The Z dimension of the preferred cluster, in blocks. Must be equal to the field of

###### ‣ z z

CUlaunchAttributeValue::clusterDim.