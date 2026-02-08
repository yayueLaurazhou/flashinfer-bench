# enum CUarray_format

Array formats

###### Values

**CU_AD_FORMAT_UNSIGNED_INT8 = 0x01**
Unsigned 8-bit integers
**CU_AD_FORMAT_UNSIGNED_INT16 = 0x02**
Unsigned 16-bit integers
**CU_AD_FORMAT_UNSIGNED_INT32 = 0x03**
Unsigned 32-bit integers
**CU_AD_FORMAT_SIGNED_INT8 = 0x08**
Signed 8-bit integers
**CU_AD_FORMAT_SIGNED_INT16 = 0x09**
Signed 16-bit integers
**CU_AD_FORMAT_SIGNED_INT32 = 0x0a**
Signed 32-bit integers
**CU_AD_FORMAT_HALF = 0x10**
16-bit floating point
**CU_AD_FORMAT_FLOAT = 0x20**
32-bit floating point


CUDA Driver API TRM-06703-001 _vRelease Version  |  17


Modules


**CU_AD_FORMAT_NV12 = 0xb0**
8-bit YUV planar format, with 4:2:0 sampling
**CU_AD_FORMAT_UNORM_INT8X1 = 0xc0**
1 channel unsigned 8-bit normalized integer
**CU_AD_FORMAT_UNORM_INT8X2 = 0xc1**
2 channel unsigned 8-bit normalized integer
**CU_AD_FORMAT_UNORM_INT8X4 = 0xc2**
4 channel unsigned 8-bit normalized integer
**CU_AD_FORMAT_UNORM_INT16X1 = 0xc3**
1 channel unsigned 16-bit normalized integer
**CU_AD_FORMAT_UNORM_INT16X2 = 0xc4**
2 channel unsigned 16-bit normalized integer
**CU_AD_FORMAT_UNORM_INT16X4 = 0xc5**
4 channel unsigned 16-bit normalized integer
**CU_AD_FORMAT_SNORM_INT8X1 = 0xc6**
1 channel signed 8-bit normalized integer
**CU_AD_FORMAT_SNORM_INT8X2 = 0xc7**
2 channel signed 8-bit normalized integer
**CU_AD_FORMAT_SNORM_INT8X4 = 0xc8**
4 channel signed 8-bit normalized integer
**CU_AD_FORMAT_SNORM_INT16X1 = 0xc9**
1 channel signed 16-bit normalized integer
**CU_AD_FORMAT_SNORM_INT16X2 = 0xca**
2 channel signed 16-bit normalized integer
**CU_AD_FORMAT_SNORM_INT16X4 = 0xcb**
4 channel signed 16-bit normalized integer
**CU_AD_FORMAT_BC1_UNORM = 0x91**
4 channel unsigned normalized block-compressed (BC1 compression) format
**CU_AD_FORMAT_BC1_UNORM_SRGB = 0x92**
4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding
**CU_AD_FORMAT_BC2_UNORM = 0x93**
4 channel unsigned normalized block-compressed (BC2 compression) format
**CU_AD_FORMAT_BC2_UNORM_SRGB = 0x94**
4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding
**CU_AD_FORMAT_BC3_UNORM = 0x95**
4 channel unsigned normalized block-compressed (BC3 compression) format
**CU_AD_FORMAT_BC3_UNORM_SRGB = 0x96**
4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding
**CU_AD_FORMAT_BC4_UNORM = 0x97**
1 channel unsigned normalized block-compressed (BC4 compression) format
**CU_AD_FORMAT_BC4_SNORM = 0x98**
1 channel signed normalized block-compressed (BC4 compression) format
**CU_AD_FORMAT_BC5_UNORM = 0x99**


CUDA Driver API TRM-06703-001 _vRelease Version  |  18


Modules


2 channel unsigned normalized block-compressed (BC5 compression) format
**CU_AD_FORMAT_BC5_SNORM = 0x9a**
2 channel signed normalized block-compressed (BC5 compression) format
**CU_AD_FORMAT_BC6H_UF16 = 0x9b**
3 channel unsigned half-float block-compressed (BC6H compression) format
**CU_AD_FORMAT_BC6H_SF16 = 0x9c**
3 channel signed half-float block-compressed (BC6H compression) format
**CU_AD_FORMAT_BC7_UNORM = 0x9d**
4 channel unsigned normalized block-compressed (BC7 compression) format
**CU_AD_FORMAT_BC7_UNORM_SRGB = 0x9e**
4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding
**CU_AD_FORMAT_P010 = 0x9f**
10-bit YUV planar format, with 4:2:0 sampling
**CU_AD_FORMAT_P016 = 0xa1**
16-bit YUV planar format, with 4:2:0 sampling
**CU_AD_FORMAT_NV16 = 0xa2**
8-bit YUV planar format, with 4:2:2 sampling
**CU_AD_FORMAT_P210 = 0xa3**
10-bit YUV planar format, with 4:2:2 sampling
**CU_AD_FORMAT_P216 = 0xa4**
16-bit YUV planar format, with 4:2:2 sampling
**CU_AD_FORMAT_YUY2 = 0xa5**
2 channel, 8-bit YUV packed planar format, with 4:2:2 sampling
**CU_AD_FORMAT_Y210 = 0xa6**
2 channel, 10-bit YUV packed planar format, with 4:2:2 sampling
**CU_AD_FORMAT_Y216 = 0xa7**
2 channel, 16-bit YUV packed planar format, with 4:2:2 sampling
**CU_AD_FORMAT_AYUV = 0xa8**
4 channel, 8-bit YUV packed planar format, with 4:4:4 sampling
**CU_AD_FORMAT_Y410 = 0xa9**
10-bit YUV packed planar format, with 4:4:4 sampling
**CU_AD_FORMAT_Y416 = 0xb1**
4 channel, 12-bit YUV packed planar format, with 4:4:4 sampling
**CU_AD_FORMAT_Y444_PLANAR8 = 0xb2**
3 channel 8-bit YUV planar format, with 4:4:4 sampling
**CU_AD_FORMAT_Y444_PLANAR10 = 0xb3**
3 channel 10-bit YUV planar format, with 4:4:4 sampling
**CU_AD_FORMAT_YUV444_8bit_SemiPlanar = 0xb4**
3 channel 8-bit YUV semi-planar format, with 4:4:4 sampling
**CU_AD_FORMAT_YUV444_16bit_SemiPlanar = 0xb5**
3 channel 16-bit YUV semi-planar format, with 4:4:4 sampling
**CU_AD_FORMAT_UNORM_INT_101010_2 = 0x50**
4 channel unorm R10G10B10A2 RGB format


CUDA Driver API TRM-06703-001 _vRelease Version  |  19


Modules


**CU_AD_FORMAT_UINT8_PACKED_422 = 0x51**
4 channel unsigned 8-bit YUV packed format, with 4:2:2 sampling
**CU_AD_FORMAT_UINT8_PACKED_444 = 0x52**
4 channel unsigned 8-bit YUV packed format, with 4:4:4 sampling
**CU_AD_FORMAT_UINT8_SEMIPLANAR_420 = 0x53**
3 channel unsigned 8-bit YUV semi-planar format, with 4:2:0 sampling
**CU_AD_FORMAT_UINT16_SEMIPLANAR_420 = 0x54**
3 channel unsigned 16-bit YUV semi-planar format, with 4:2:0 sampling
**CU_AD_FORMAT_UINT8_SEMIPLANAR_422 = 0x55**
3 channel unsigned 8-bit YUV semi-planar format, with 4:2:2 sampling
**CU_AD_FORMAT_UINT16_SEMIPLANAR_422 = 0x56**
3 channel unsigned 16-bit YUV semi-planar format, with 4:2:2 sampling
**CU_AD_FORMAT_UINT8_SEMIPLANAR_444 = 0x57**
3 channel unsigned 8-bit YUV semi-planar format, with 4:4:4 sampling
**CU_AD_FORMAT_UINT16_SEMIPLANAR_444 = 0x58**
3 channel unsigned 16-bit YUV semi-planar format, with 4:4:4 sampling
**CU_AD_FORMAT_UINT8_PLANAR_420 = 0x59**
3 channel unsigned 8-bit YUV planar format, with 4:2:0 sampling
**CU_AD_FORMAT_UINT16_PLANAR_420 = 0x5a**
3 channel unsigned 16-bit YUV planar format, with 4:2:0 sampling
**CU_AD_FORMAT_UINT8_PLANAR_422 = 0x5b**
3 channel unsigned 8-bit YUV planar format, with 4:2:2 sampling
**CU_AD_FORMAT_UINT16_PLANAR_422 = 0x5c**
3 channel unsigned 16-bit YUV planar format, with 4:2:2 sampling
**CU_AD_FORMAT_UINT8_PLANAR_444 = 0x5d**
3 channel unsigned 8-bit YUV planar format, with 4:4:4 sampling
**CU_AD_FORMAT_UINT16_PLANAR_444 = 0x5e**
3 channel unsigned 16-bit YUV planar format, with 4:4:4 sampling
**CU_AD_FORMAT_MAX = 0x7FFFFFFF**