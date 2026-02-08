# 19.1. Nearest-Point Sampling

## 19.1. Nearest-Point Sampling[](#nearest-point-sampling "Permalink to this headline")

In this filtering mode, the value returned by the texture fetch is

* *tex(x)=T[i]* for a one-dimensional texture,
* *tex(x,y)=T[i,j]* for a two-dimensional texture,
* *tex(x,y,z)=T[i,j,k]* for a three-dimensional texture,

where *i=floor(x)*, *j=floor(y)*, and *k=floor(z)*.

[Figure 36](#nearest-point-sampling-nearest-point-sampling-fig) illustrates nearest-point sampling for a one-dimensional texture with *N=4*.

![_images/nearest-point-sampling-of-1-d-texture-of-4-texels.png](_images/nearest-point-sampling-of-1-d-texture-of-4-texels.png)


Figure 36 Nearest-Point Sampling Filtering Mode[](#nearest-point-sampling-nearest-point-sampling-fig "Permalink to this image")

For integer textures, the value returned by the texture fetch can be optionally remapped to [0.0, 1.0] (see [Texture Memory](#texture-memory)).