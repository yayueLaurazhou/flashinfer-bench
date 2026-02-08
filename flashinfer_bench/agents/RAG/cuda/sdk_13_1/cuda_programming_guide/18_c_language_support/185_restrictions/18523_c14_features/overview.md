# 18.5.23. C++14 Features

### 18.5.23. C++14 Features[](#c-14-features "Permalink to this headline")

C++14 features enabled by default by the host compiler are also supported by nvcc. Passing nvcc `-std=c++14` flag turns on all C++14 features and also invokes the host preprocessor, compiler and linker with the corresponding C++14 dialect option [20](#fn27). This section describes the restrictions on the supported C++14 features.