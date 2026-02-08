# 18.5.22. C++11 Features

### 18.5.22. C++11 Features[](#c-11-features "Permalink to this headline")

C++11 features that are enabled by default by the host compiler are also supported by nvcc, subject to the restrictions described in this document. In addition, invoking nvcc with `-std=c++11` flag turns on all C++11 features and also invokes the host preprocessor, compiler and linker with the corresponding C++11 dialect option [14](#fn21).