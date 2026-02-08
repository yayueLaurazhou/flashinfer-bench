# 17. Mathematical Functions

# 17. Mathematical Functions[](#mathematical-functions-appendix "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

The reference manual lists, along with their description, all the functions of the C/C++ standard library mathematical functions that are supported in device code, as well as all intrinsic functions (that are only supported in device code).

This section provides accuracy information for some of these functions when applicable. It uses ULP for quantification. For further information on the definition of the Unit in the Last Place (ULP), please see Jean-Michel Muller’s paper *On the definition of ulp(x)*, RR-5504, LIP RR-2005-09, INRIA, LIP. 2005, pp.16 at <https://hal.inria.fr/inria-00070503/document>.

Mathematical functions supported in device code do not set the global `errno` variable, nor report any floating-point exceptions to indicate errors; thus, if error diagnostic mechanisms are required, the user should implement additional screening for inputs and outputs of the functions. The user is responsible for the validity of pointer arguments. The user must not pass uninitialized parameters to the Mathematical functions as this may result in undefined behavior: functions are inlined in the user program and thus are subject to compiler optimizations.