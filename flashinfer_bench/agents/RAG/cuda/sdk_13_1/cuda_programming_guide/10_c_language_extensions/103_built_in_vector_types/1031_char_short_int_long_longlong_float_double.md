# 10.3.1. char, short, int, long, longlong, float, double

### 10.3.1. char, short, int, long, longlong, float, double[](#char-short-int-long-longlong-float-double "Permalink to this headline")

These are vector types derived from the basic integer and floating-point types. They are structures and the 1st, 2nd, 3rd, and 4th components are accessible through the fields `x`, `y`, `z`, and `w`, respectively. They all come with a constructor function of the form `make_<type name>`; for example,

```
int2 make_int2(int x, int y);
```

which creates a vector of type `int2` with value`(x, y)`.

The alignment requirements of the vector types are detailed in [Table 7](#vector-types-alignment-requirements-in-device-code).

Table 7 Alignment Requirements[](#vector-types-alignment-requirements-in-device-code "Permalink to this table")




| Type | Alignment |
| --- | --- |
| char1, uchar1 | 1 |
| char2, uchar2 | 2 |
| char3, uchar3 | 1 |
| char4, uchar4 | 4 |
| short1, ushort1 | 2 |
| short2, ushort2 | 4 |
| short3, ushort3 | 2 |
| short4, ushort4 | 8 |
| int1, uint1 | 4 |
| int2, uint2 | 8 |
| int3, uint3 | 4 |
| int4, uint4 | 16 |
| long1, ulong1 | 4 if sizeof(long) is equal to sizeof(int) 8, otherwise |
| long2, ulong2 | 8 if sizeof(long) is equal to sizeof(int), 16, otherwise |
| long3, ulong3 | 4 if sizeof(long) is equal to sizeof(int), 8, otherwise |
| long4 [3](#fn32a) | 16 |
| long4\_16a |
| long4\_32a | 32 |
| ulong4 [3](#fn32a) | 16 |
| ulong4\_16a |
| ulong4\_32a | 32 |
| longlong1, ulonglong1 | 8 |
| longlong2, ulonglong2 | 16 |
| longlong3, ulonglong3 | 8 |
| longlong4 [3](#fn32a) | 16 |
| longlong4\_16a |
| longlong4\_32a | 32 |
| ulonglong4 [3](#fn32a) | 16 |
| ulonglong4\_16a |
| ulonglong4\_32a | 32 |
| float1 | 4 |
| float2 | 8 |
| float3 | 4 |
| float4 | 16 |
| double1 | 8 |
| double2 | 16 |
| double3 | 8 |
| double4 [3](#fn32a) | 16 |
| double4\_16a |
| double4\_32a | 32 |

3([1](#id155),[2](#id156),[3](#id157),[4](#id158),[5](#id159))
:   This vector type was deprecated in CUDA Toolkit 13.0. Please use the `_16a` or `_32a` variant instead, depending on your alignment requirements.