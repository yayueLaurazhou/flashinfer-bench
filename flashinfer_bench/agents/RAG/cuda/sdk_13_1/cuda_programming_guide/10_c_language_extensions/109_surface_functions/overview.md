# 10.9. Surface Functions

## 10.9. Surface Functions[](#surface-functions "Permalink to this headline")

Surface functions are only supported by devices of compute capability 2.0 and higher.

Surface objects are described in described in [Surface Object API](#surface-object-api-appendix).

In the sections below, `boundaryMode` specifies the boundary mode, that is how out-of-range surface coordinates are handled; it is equal to either `cudaBoundaryModeClamp`, in which case out-of-range coordinates are clamped to the valid range, or `cudaBoundaryModeZero`, in which case out-of-range reads return zero and out-of-range writes are ignored, or `cudaBoundaryModeTrap`, in which case out-of-range accesses cause the kernel execution to fail.