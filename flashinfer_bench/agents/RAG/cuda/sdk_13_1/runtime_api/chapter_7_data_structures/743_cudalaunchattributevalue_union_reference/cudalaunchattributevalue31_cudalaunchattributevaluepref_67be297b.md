# cudaLaunchAttributeValue::@31 cudaLaunchAttributeValue::preferredClusterDim

Value of launch attribute cudaLaunchAttributePreferredClusterDimension that represents the desired
preferred cluster dimensions for the kernel. Opaque type with the following fields:


CUDA Runtime API vRelease Version  |  630


Data Structures


   - The X dimension of the preferred cluster, in blocks. Must be a divisor of the grid X dimension,

##### ‣ x

and must be a multiple of the x field of cudaLaunchAttributeValue::clusterDim.
   - The Y dimension of the preferred cluster, in blocks. Must be a divisor of the grid Y dimension,

##### ‣ y

and must be a multiple of the y field of cudaLaunchAttributeValue::clusterDim.
   - The Z dimension of the preferred cluster, in blocks. Must be equal to the field of

##### ‣ z z

cudaLaunchAttributeValue::clusterDim.