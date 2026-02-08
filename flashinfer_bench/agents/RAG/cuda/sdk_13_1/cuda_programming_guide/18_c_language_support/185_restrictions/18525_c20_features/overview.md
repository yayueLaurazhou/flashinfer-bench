# 18.5.25. C++20 Features

### 18.5.25. C++20 Features[](#c-20-features "Permalink to this headline")

C++20 features enabled by default by the host compiler are also supported by nvcc. Passing nvcc `-std=c++20` flag turns on all C++20 features and also invokes the host preprocessor, compiler and linker with the corresponding C++20 dialect option [22](#fn29). This section describes the restrictions on the supported C++20 features.