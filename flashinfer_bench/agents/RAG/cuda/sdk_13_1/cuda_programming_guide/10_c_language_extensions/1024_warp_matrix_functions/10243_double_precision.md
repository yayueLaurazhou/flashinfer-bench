# 10.24.3. Double Precision

### 10.24.3. Double Precision[](#double-precision "Permalink to this headline")

Tensor Cores support double-precision floating point operations on devices with compute capability 8.0 and higher. To use this new functionality, a `fragment` with the `double` type must be used. The `mma_sync` operation will be performed with the .rn (rounds to nearest even) rounding modifier.