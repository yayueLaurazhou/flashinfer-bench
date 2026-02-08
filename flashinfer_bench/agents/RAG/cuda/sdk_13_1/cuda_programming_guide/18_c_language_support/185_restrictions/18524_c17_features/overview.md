# 18.5.24. C++17 Features

### 18.5.24. C++17 Features[](#c-17-features "Permalink to this headline")

C++17 features enabled by default by the host compiler are also supported by nvcc. Passing nvcc `-std=c++17` flag turns on all C++17 features and also invokes the host preprocessor, compiler and linker with the corresponding C++17 dialect option [21](#fn28). This section describes the restrictions on the supported C++17 features.