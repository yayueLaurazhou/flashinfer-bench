# 10.24. Warp Matrix Functions

## 10.24. Warp Matrix Functions[](#warp-matrix-functions "Permalink to this headline")

C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form `D=A*B+C`. These operations are supported on mixed-precision floating point data for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a [warp](#simt-architecture). In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire [warp](#simt-architecture), otherwise the code execution is likely to hang.