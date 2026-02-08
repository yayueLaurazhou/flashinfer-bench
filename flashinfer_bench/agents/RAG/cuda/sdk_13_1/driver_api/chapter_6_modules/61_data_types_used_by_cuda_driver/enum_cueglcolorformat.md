# enum CUeglColorFormat

CUDA EGL Color Format - The different planar and multiplanar formats currently
supported for CUDA_EGL interops. Three channel formats are currently not supported for
CU_EGL_FRAME_TYPE_ARRAY

###### Values

**CU_EGL_COLOR_FORMAT_YUV420_PLANAR = 0x00**


CUDA Driver API TRM-06703-001 _vRelease Version  |  32


Modules


Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR = 0x01**
Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as
YUV420Planar.
**CU_EGL_COLOR_FORMAT_YUV422_PLANAR = 0x02**
Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR = 0x03**
Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar.
**CU_EGL_COLOR_FORMAT_RGB = 0x04**
R/G/B three channels in one surface with BGR byte ordering. Only pitch linear format supported.
**CU_EGL_COLOR_FORMAT_BGR = 0x05**
R/G/B three channels in one surface with RGB byte ordering. Only pitch linear format supported.
**CU_EGL_COLOR_FORMAT_ARGB = 0x06**
R/G/B/A four channels in one surface with BGRA byte ordering.
**CU_EGL_COLOR_FORMAT_RGBA = 0x07**
R/G/B/A four channels in one surface with ABGR byte ordering.
**CU_EGL_COLOR_FORMAT_L = 0x08**
single luminance channel in one surface.
**CU_EGL_COLOR_FORMAT_R = 0x09**
single color channel in one surface.
**CU_EGL_COLOR_FORMAT_YUV444_PLANAR = 0x0A**
Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR = 0x0B**
Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as
YUV444Planar.
**CU_EGL_COLOR_FORMAT_YUYV_422 = 0x0C**
Y, U, V in one surface, interleaved as UYVY in one channel.
**CU_EGL_COLOR_FORMAT_UYVY_422 = 0x0D**
Y, U, V in one surface, interleaved as YUYV in one channel.
**CU_EGL_COLOR_FORMAT_ABGR = 0x0E**
R/G/B/A four channels in one surface with RGBA byte ordering.
**CU_EGL_COLOR_FORMAT_BGRA = 0x0F**
R/G/B/A four channels in one surface with ARGB byte ordering.
**CU_EGL_COLOR_FORMAT_A = 0x10**
Alpha color format - one channel in one surface.
**CU_EGL_COLOR_FORMAT_RG = 0x11**
R/G color format - two channels in one surface with GR byte ordering
**CU_EGL_COLOR_FORMAT_AYUV = 0x12**
Y, U, V, A four channels in one surface, interleaved as VUYA.
**CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR = 0x13**
Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V
height = Y height.


CUDA Driver API TRM-06703-001 _vRelease Version  |  33


Modules


**CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR = 0x14**
Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V
height = Y height.
**CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR = 0x15**
Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR = 0x16**
Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/
V height = Y height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR = 0x17**
Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y
width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR = 0x18**
Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/
V height = Y height.
**CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR = 0x19**
Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y
width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_VYUY_ER = 0x1A**
Extended Range Y, U, V in one surface, interleaved as YVYU in one channel.
**CU_EGL_COLOR_FORMAT_UYVY_ER = 0x1B**
Extended Range Y, U, V in one surface, interleaved as YUYV in one channel.
**CU_EGL_COLOR_FORMAT_YUYV_ER = 0x1C**
Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.
**CU_EGL_COLOR_FORMAT_YVYU_ER = 0x1D**
Extended Range Y, U, V in one surface, interleaved as VYUY in one channel.
**CU_EGL_COLOR_FORMAT_YUV_ER = 0x1E**
Extended Range Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear
format supported.
**CU_EGL_COLOR_FORMAT_YUVA_ER = 0x1F**
Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY.
**CU_EGL_COLOR_FORMAT_AYUV_ER = 0x20**
Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA.
**CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER = 0x21**
Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER = 0x22**
Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER = 0x23**
Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER = 0x24**
Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y
width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER = 0x25**


CUDA Driver API TRM-06703-001 _vRelease Version  |  34


Modules


Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width =
1/2 Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER = 0x26**
Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width =
1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER = 0x27**
Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER = 0x28**
Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER = 0x29**
Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER = 0x2A**
Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y
width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER = 0x2B**
Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width =
1/2 Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER = 0x2C**
Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width =
1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_BAYER_RGGB = 0x2D**
Bayer format - one channel in one surface with interleaved RGGB ordering.
**CU_EGL_COLOR_FORMAT_BAYER_BGGR = 0x2E**
Bayer format - one channel in one surface with interleaved BGGR ordering.
**CU_EGL_COLOR_FORMAT_BAYER_GRBG = 0x2F**
Bayer format - one channel in one surface with interleaved GRBG ordering.
**CU_EGL_COLOR_FORMAT_BAYER_GBRG = 0x30**
Bayer format - one channel in one surface with interleaved GBRG ordering.
**CU_EGL_COLOR_FORMAT_BAYER10_RGGB = 0x31**
Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER10_BGGR = 0x32**
Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER10_GRBG = 0x33**
Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER10_GBRG = 0x34**
Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER12_RGGB = 0x35**
Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12
bits used 4 bits No-op.


CUDA Driver API TRM-06703-001 _vRelease Version  |  35


Modules


**CU_EGL_COLOR_FORMAT_BAYER12_BGGR = 0x36**
Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER12_GRBG = 0x37**
Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER12_GBRG = 0x38**
Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER14_RGGB = 0x39**
Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14
bits used 2 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER14_BGGR = 0x3A**
Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14
bits used 2 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER14_GRBG = 0x3B**
Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14
bits used 2 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER14_GBRG = 0x3C**
Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14
bits used 2 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER20_RGGB = 0x3D**
Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20
bits used 12 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER20_BGGR = 0x3E**
Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20
bits used 12 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER20_GRBG = 0x3F**
Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20
bits used 12 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER20_GBRG = 0x40**
Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20
bits used 12 bits No-op.
**CU_EGL_COLOR_FORMAT_YVU444_PLANAR = 0x41**
Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_YVU422_PLANAR = 0x42**
Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y
height.
**CU_EGL_COLOR_FORMAT_YVU420_PLANAR = 0x43**
Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB = 0x44**


CUDA Driver API TRM-06703-001 _vRelease Version  |  36


Modules


Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering
and mapped to opaque integer datatype.
**CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR = 0x45**
Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering
and mapped to opaque integer datatype.
**CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG = 0x46**
Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering
and mapped to opaque integer datatype.
**CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG = 0x47**
Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering
and mapped to opaque integer datatype.
**CU_EGL_COLOR_FORMAT_BAYER_BCCR = 0x48**
Bayer format - one channel in one surface with interleaved BCCR ordering.
**CU_EGL_COLOR_FORMAT_BAYER_RCCB = 0x49**
Bayer format - one channel in one surface with interleaved RCCB ordering.
**CU_EGL_COLOR_FORMAT_BAYER_CRBC = 0x4A**
Bayer format - one channel in one surface with interleaved CRBC ordering.
**CU_EGL_COLOR_FORMAT_BAYER_CBRC = 0x4B**
Bayer format - one channel in one surface with interleaved CBRC ordering.
**CU_EGL_COLOR_FORMAT_BAYER10_CCCC = 0x4C**
Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER12_BCCR = 0x4D**
Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER12_RCCB = 0x4E**
Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER12_CRBC = 0x4F**
Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER12_CBRC = 0x50**
Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**CU_EGL_COLOR_FORMAT_BAYER12_CCCC = 0x51**
Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**CU_EGL_COLOR_FORMAT_Y = 0x52**
Color format for single Y plane.
**CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020 = 0x53**
Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020 = 0x54**
Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


CUDA Driver API TRM-06703-001 _vRelease Version  |  37


Modules


**CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020 = 0x55**
Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height= 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020 = 0x56**
Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709 = 0x57**
Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709 = 0x58**
Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709 = 0x59**
Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709 = 0x5A**
Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709 = 0x5B**
Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020 = 0x5C**
Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020 = 0x5D**
Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR = 0x5E**
Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709 = 0x5F**
Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.
**CU_EGL_COLOR_FORMAT_Y_ER = 0x60**
Extended Range Color format for single Y plane.
**CU_EGL_COLOR_FORMAT_Y_709_ER = 0x61**
Extended Range Color format for single Y plane.
**CU_EGL_COLOR_FORMAT_Y10_ER = 0x62**
Extended Range Color format for single Y10 plane.
**CU_EGL_COLOR_FORMAT_Y10_709_ER = 0x63**
Extended Range Color format for single Y10 plane.
**CU_EGL_COLOR_FORMAT_Y12_ER = 0x64**
Extended Range Color format for single Y12 plane.
**CU_EGL_COLOR_FORMAT_Y12_709_ER = 0x65**
Extended Range Color format for single Y12 plane.
**CU_EGL_COLOR_FORMAT_YUVA = 0x66**
Y, U, V, A four channels in one surface, interleaved as AVUY.
**CU_EGL_COLOR_FORMAT_YUV = 0x67**
Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported.
**CU_EGL_COLOR_FORMAT_YVYU = 0x68**
Y, U, V in one surface, interleaved as YVYU in one channel.
**CU_EGL_COLOR_FORMAT_VYUY = 0x69**


CUDA Driver API TRM-06703-001 _vRelease Version  |  38


Modules


Y, U, V in one surface, interleaved as VYUY in one channel.
**CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER = 0x6A**
Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER = 0x6B**
Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER = 0x6C**
Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V
height = Y height.
**CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER = 0x6D**
Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V
height = Y height.
**CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER = 0x6E**
Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER = 0x6F**
Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER = 0x70**
Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V
height = Y height.
**CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER = 0x71**
Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V
height = Y height.
**CU_EGL_COLOR_FORMAT_UYVY_709 = 0x72**
Y, U, V in one surface, interleaved as UYVY in one channel.
**CU_EGL_COLOR_FORMAT_UYVY_709_ER = 0x73**
Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.
**CU_EGL_COLOR_FORMAT_UYVY_2020 = 0x74**
Y, U, V in one surface, interleaved as UYVY in one channel.
**CU_EGL_COLOR_FORMAT_MAX**