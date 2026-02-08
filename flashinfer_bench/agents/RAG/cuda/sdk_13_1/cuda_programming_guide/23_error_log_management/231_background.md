# 23.1. Background

## 23.1. Background[](#id446 "Permalink to this headline")

Traditionally, the only indication of a failed CUDA API call is the return of a non-zero code.
As of CUDA Toolkit 12.9, the CUDA Runtime defines over 100 different return codes
for error conditions, but many of them are generic and give the developer no assistance with debugging the cause.