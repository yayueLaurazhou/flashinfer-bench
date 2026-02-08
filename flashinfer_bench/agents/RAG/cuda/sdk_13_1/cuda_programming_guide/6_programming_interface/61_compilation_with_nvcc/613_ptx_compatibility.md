# 6.1.3. PTX Compatibility

### 6.1.3. PTX Compatibility[](#ptx-compatibility "Permalink to this headline")

Some *PTX* instructions are only supported on devices of higher compute capabilities. For example, [Warp Shuffle Functions](#warp-shuffle-functions) are only supported on devices of compute capability 5.0 and above. The `-arch` compiler option specifies the compute capability that is assumed when compiling C++ to *PTX* code. So, code that contains warp shuffle, for example, must be compiled with `-arch=compute_50` (or higher).

*PTX* code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability. Note that a binary compiled from an earlier PTX version may not make use of some hardware features. For example, a binary targeting devices of compute capability 7.0 (Volta) compiled from PTX generated for compute capability 6.0 (Pascal) will not make use of Tensor Core instructions, since these were not available on Pascal. As a result, the final binary may perform worse than would be possible if the binary were generated using the latest version of PTX.

*PTX* code compiled to target [Architecture-Specific Features](#architecture-specific-features) only runs on the exact same physical architecture and nowhere else. Architecture-specific *PTX* code is not forward and backward compatible.
Example code compiled with `sm_90a` or `compute_90a` only runs on devices with compute capability 9.0 and is not backward or forward compatible.

*PTX* code compiled to target [Family-Specific Features](#family-specific-features) only runs on the exact same physical architecture and other architectures in the same family. Family-specific *PTX* code is forward compatible with other devices in the same family, and is not backward compatible.
Example code compiled with `sm_100f` or `compute_100f` only runs on devices with compute capability 10.0 and 10.3. [Table 25](#family-specific-compatibility) shows the compatibility of family-specific targets with compute capability.