# 18. C++ Language Support

# 18. C++ Language Support[](#c-language-support "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

As described in [Compilation with NVCC](#compilation-with-nvcc), CUDA source files compiled with `nvcc` can include a mix of host code and device code. The CUDA front-end compiler aims to emulate the host compiler behavior with respect to C++ input code. The input source code is processed according to the C++ ISO/IEC 14882:2003, C++ ISO/IEC 14882:2011, C++ ISO/IEC 14882:2014 or C++ ISO/IEC 14882:2017 specifications, and the CUDA front-end compiler aims to emulate any host compiler divergences from the ISO specification. In addition, the supported language is extended with CUDA-specific constructs described in this document [6](#fn13), and is subject to the restrictions described below.

[C++11 Language Features](#cpp11-language-features), [C++14 Language Features](#cpp14-language-features) and [C++17 Language Features](#cpp17-language-features) provide support matrices for the C++11, C++14, C++17 and C++20 features, respectively. [Restrictions](#language-restrictions) lists the language restrictions. [Polymorphic Function Wrappers](#polymorphic-function-wrappers) and [Extended Lambdas](#extended-lambda) describe additional features. [Code Samples](#code-samples) gives code samples.