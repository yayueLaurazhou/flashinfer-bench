# 10.21. Warp Reduce Functions

## 10.21. Warp Reduce Functions[](#warp-reduce-functions "Permalink to this headline")

The `__reduce_sync(unsigned mask, T value)` intrinsics perform a reduction operation on the data provided in `value` after synchronizing threads named in `mask`. T can be unsigned or signed for {add, min, max} and unsigned only for {and, or, xor} operations.

Supported by devices of compute capability 8.x or higher.