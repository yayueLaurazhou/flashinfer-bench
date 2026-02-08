# int cudaFuncAttributes::ptxVersion

The PTX virtual architecture version for which the function was compiled. This value is the major PTX
version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13.