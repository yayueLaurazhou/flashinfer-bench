# enum cudaEglColorFormat

CUDA EGL Color Format - The different planar and multiplanar formats currently supported for
CUDA_EGL interops.

##### Values

**cudaEglColorFormatYUV420Planar = 0**
Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**cudaEglColorFormatYUV420SemiPlanar = 1**
Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as
YUV420Planar.
**cudaEglColorFormatYUV422Planar = 2**
Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.
**cudaEglColorFormatYUV422SemiPlanar = 3**
Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar.
**cudaEglColorFormatARGB = 6**
R/G/B/A four channels in one surface with BGRA byte ordering.
**cudaEglColorFormatRGBA = 7**


CUDA Runtime API vRelease Version  |  536


Modules


R/G/B/A four channels in one surface with ABGR byte ordering.
**cudaEglColorFormatL = 8**
single luminance channel in one surface.
**cudaEglColorFormatR = 9**
single color channel in one surface.
**cudaEglColorFormatYUV444Planar = 10**
Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.
**cudaEglColorFormatYUV444SemiPlanar = 11**
Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as
YUV444Planar.
**cudaEglColorFormatYUYV422 = 12**
Y, U, V in one surface, interleaved as UYVY in one channel.
**cudaEglColorFormatUYVY422 = 13**
Y, U, V in one surface, interleaved as YUYV in one channel.
**cudaEglColorFormatABGR = 14**
R/G/B/A four channels in one surface with RGBA byte ordering.
**cudaEglColorFormatBGRA = 15**
R/G/B/A four channels in one surface with ARGB byte ordering.
**cudaEglColorFormatA = 16**
Alpha color format - one channel in one surface.
**cudaEglColorFormatRG = 17**
R/G color format - two channels in one surface with GR byte ordering
**cudaEglColorFormatAYUV = 18**
Y, U, V, A four channels in one surface, interleaved as VUYA.
**cudaEglColorFormatYVU444SemiPlanar = 19**
Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V
height = Y height.
**cudaEglColorFormatYVU422SemiPlanar = 20**
Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V
height = Y height.
**cudaEglColorFormatYVU420SemiPlanar = 21**
Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**cudaEglColorFormatY10V10U10_444SemiPlanar = 22**
Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/
V height = Y height.
**cudaEglColorFormatY10V10U10_420SemiPlanar = 23**
Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y
width, U/V height = 1/2 Y height.
**cudaEglColorFormatY12V12U12_444SemiPlanar = 24**
Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/
V height = Y height.
**cudaEglColorFormatY12V12U12_420SemiPlanar = 25**


CUDA Runtime API vRelease Version  |  537


Modules


Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y
width, U/V height = 1/2 Y height.
**cudaEglColorFormatVYUY_ER = 26**
Extended Range Y, U, V in one surface, interleaved as YVYU in one channel.
**cudaEglColorFormatUYVY_ER = 27**
Extended Range Y, U, V in one surface, interleaved as YUYV in one channel.
**cudaEglColorFormatYUYV_ER = 28**
Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.
**cudaEglColorFormatYVYU_ER = 29**
Extended Range Y, U, V in one surface, interleaved as VYUY in one channel.
**cudaEglColorFormatYUVA_ER = 31**
Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY.
**cudaEglColorFormatAYUV_ER = 32**
Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA.
**cudaEglColorFormatYUV444Planar_ER = 33**
Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height.
**cudaEglColorFormatYUV422Planar_ER = 34**
Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.
**cudaEglColorFormatYUV420Planar_ER = 35**
Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**cudaEglColorFormatYUV444SemiPlanar_ER = 36**
Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y
width, U/V height = Y height.
**cudaEglColorFormatYUV422SemiPlanar_ER = 37**
Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width =
1/2 Y width, U/V height = Y height.
**cudaEglColorFormatYUV420SemiPlanar_ER = 38**
Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width =
1/2 Y width, U/V height = 1/2 Y height.
**cudaEglColorFormatYVU444Planar_ER = 39**
Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height.
**cudaEglColorFormatYVU422Planar_ER = 40**
Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.
**cudaEglColorFormatYVU420Planar_ER = 41**
Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**cudaEglColorFormatYVU444SemiPlanar_ER = 42**
Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y
width, U/V height = Y height.
**cudaEglColorFormatYVU422SemiPlanar_ER = 43**
Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width =
1/2 Y width, U/V height = Y height.
**cudaEglColorFormatYVU420SemiPlanar_ER = 44**


CUDA Runtime API vRelease Version  |  538


Modules


Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width =
1/2 Y width, U/V height = 1/2 Y height.
**cudaEglColorFormatBayerRGGB = 45**
Bayer format - one channel in one surface with interleaved RGGB ordering.
**cudaEglColorFormatBayerBGGR = 46**
Bayer format - one channel in one surface with interleaved BGGR ordering.
**cudaEglColorFormatBayerGRBG = 47**
Bayer format - one channel in one surface with interleaved GRBG ordering.
**cudaEglColorFormatBayerGBRG = 48**
Bayer format - one channel in one surface with interleaved GBRG ordering.
**cudaEglColorFormatBayer10RGGB = 49**
Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**cudaEglColorFormatBayer10BGGR = 50**
Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**cudaEglColorFormatBayer10GRBG = 51**
Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**cudaEglColorFormatBayer10GBRG = 52**
Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**cudaEglColorFormatBayer12RGGB = 53**
Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatBayer12BGGR = 54**
Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatBayer12GRBG = 55**
Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatBayer12GBRG = 56**
Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatBayer14RGGB = 57**
Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14
bits used 2 bits No-op.
**cudaEglColorFormatBayer14BGGR = 58**
Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14
bits used 2 bits No-op.
**cudaEglColorFormatBayer14GRBG = 59**
Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14
bits used 2 bits No-op.


CUDA Runtime API vRelease Version  |  539


Modules


**cudaEglColorFormatBayer14GBRG = 60**
Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14
bits used 2 bits No-op.
**cudaEglColorFormatBayer20RGGB = 61**
Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20
bits used 12 bits No-op.
**cudaEglColorFormatBayer20BGGR = 62**
Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20
bits used 12 bits No-op.
**cudaEglColorFormatBayer20GRBG = 63**
Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20
bits used 12 bits No-op.
**cudaEglColorFormatBayer20GBRG = 64**
Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20
bits used 12 bits No-op.
**cudaEglColorFormatYVU444Planar = 65**
Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.
**cudaEglColorFormatYVU422Planar = 66**
Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y
height.
**cudaEglColorFormatYVU420Planar = 67**
Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**cudaEglColorFormatBayerIspRGGB = 68**
Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering
and mapped to opaque integer datatype.
**cudaEglColorFormatBayerIspBGGR = 69**
Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering
and mapped to opaque integer datatype.
**cudaEglColorFormatBayerIspGRBG = 70**
Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering
and mapped to opaque integer datatype.
**cudaEglColorFormatBayerIspGBRG = 71**
Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering
and mapped to opaque integer datatype.
**cudaEglColorFormatBayerBCCR = 72**
Bayer format - one channel in one surface with interleaved BCCR ordering.
**cudaEglColorFormatBayerRCCB = 73**
Bayer format - one channel in one surface with interleaved RCCB ordering.
**cudaEglColorFormatBayerCRBC = 74**
Bayer format - one channel in one surface with interleaved CRBC ordering.
**cudaEglColorFormatBayerCBRC = 75**
Bayer format - one channel in one surface with interleaved CBRC ordering.


CUDA Runtime API vRelease Version  |  540


Modules


**cudaEglColorFormatBayer10CCCC = 76**
Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10
bits used 6 bits No-op.
**cudaEglColorFormatBayer12BCCR = 77**
Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatBayer12RCCB = 78**
Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatBayer12CRBC = 79**
Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatBayer12CBRC = 80**
Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatBayer12CCCC = 81**
Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12
bits used 4 bits No-op.
**cudaEglColorFormatY = 82**
Color format for single Y plane.
**cudaEglColorFormatYUV420SemiPlanar_2020 = 83**
Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**cudaEglColorFormatYVU420SemiPlanar_2020 = 84**
Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**cudaEglColorFormatYUV420Planar_2020 = 85**
Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**cudaEglColorFormatYVU420Planar_2020 = 86**
Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**cudaEglColorFormatYUV420SemiPlanar_709 = 87**
Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**cudaEglColorFormatYVU420SemiPlanar_709 = 88**
Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.
**cudaEglColorFormatYUV420Planar_709 = 89**
Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**cudaEglColorFormatYVU420Planar_709 = 90**
Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**cudaEglColorFormatY10V10U10_420SemiPlanar_709 = 91**
Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y
height.


CUDA Runtime API vRelease Version  |  541


Modules


**cudaEglColorFormatY10V10U10_420SemiPlanar_2020 = 92**
Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y
height.
**cudaEglColorFormatY10V10U10_422SemiPlanar_2020 = 93**
Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y
height.
**cudaEglColorFormatY10V10U10_422SemiPlanar = 94**
Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y
height.
**cudaEglColorFormatY10V10U10_422SemiPlanar_709 = 95**
Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y
height.
**cudaEglColorFormatY_ER = 96**
Extended Range Color format for single Y plane.
**cudaEglColorFormatY_709_ER = 97**
Extended Range Color format for single Y plane.
**cudaEglColorFormatY10_ER = 98**
Extended Range Color format for single Y10 plane.
**cudaEglColorFormatY10_709_ER = 99**
Extended Range Color format for single Y10 plane.
**cudaEglColorFormatY12_ER = 100**
Extended Range Color format for single Y12 plane.
**cudaEglColorFormatY12_709_ER = 101**
Extended Range Color format for single Y12 plane.
**cudaEglColorFormatYUVA = 102**
Y, U, V, A four channels in one surface, interleaved as AVUY.
**cudaEglColorFormatYVYU = 104**
Y, U, V in one surface, interleaved as YVYU in one channel.
**cudaEglColorFormatVYUY = 105**
Y, U, V in one surface, interleaved as VYUY in one channel.
**cudaEglColorFormatY10V10U10_420SemiPlanar_ER = 106**
Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER = 107**
Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**cudaEglColorFormatY10V10U10_444SemiPlanar_ER = 108**
Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V
height = Y height.
**cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER = 109**
Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V
height = Y height.
**cudaEglColorFormatY12V12U12_420SemiPlanar_ER = 110**


CUDA Runtime API vRelease Version  |  542


Modules


Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER = 111**
Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V
height = 1/2 Y height.
**cudaEglColorFormatY12V12U12_444SemiPlanar_ER = 112**
Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V
height = Y height.
**cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER = 113**
Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V
height = Y height.
**cudaEglColorFormatUYVY709 = 114**
Y, U, V in one surface, interleaved as UYVY in one channel.
**cudaEglColorFormatUYVY709_ER = 115**
Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.
**cudaEglColorFormatUYVY2020 = 116**
Y, U, V in one surface, interleaved as UYVY in one channel.