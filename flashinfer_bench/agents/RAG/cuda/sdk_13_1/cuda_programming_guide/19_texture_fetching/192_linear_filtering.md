# 19.2. Linear Filtering

## 19.2. Linear Filtering[](#linear-filtering "Permalink to this headline")

In this filtering mode, which is only available for floating-point textures, the value returned by the texture fetch is

* \(tex(x)=(1-\alpha)T[i]+{\alpha}T[i+1]\) for a one-dimensional texture,
* \(tex(x)=(1-\alpha)T[i]+{\alpha}T[i+1]\) for a one-dimensional texture,
* \(tex(x,y)=(1-\alpha)(1-\beta)T[i,j]+\alpha(1-\beta)T[i+1,j]+(1-\alpha){\beta}T[i,j+1]+\alpha{\beta}T[i+1,j+1]\) for a two-dimensional texture,
* \(tex(x,y,z)\) =

  \((1-\alpha)(1-\beta)(1-\gamma)T[i,j,k]+\alpha(1-\beta)(1-\gamma)T[i+1,j,k]+\)

  \((1-\alpha)\beta(1-\gamma)T[i,j+1,k]+\alpha\beta(1-\gamma)T[i+1,j+1,k]+\)

  \((1-\alpha)(1-\beta){\gamma}T[i,j,k+1]+\alpha(1-\beta){\gamma}T[i+1,j,k+1]+\)

  \((1-\alpha)\beta{\gamma}T[i,j+1,k+1]+\alpha\beta{\gamma}T[i+1,j+1,k+1]\)

  for a three-dimensional texture,

where:

* \(i=floor(x\ B)\*, \alpha=frac(x\ B)\*, \*x\ B\ =x-0.5,\)
* \(j=floor(y\ B)\*, \beta=frac(y\ B)\*, \*y\ B\ =y-0.5,\)
* \(k=floor(z\ B)\*, \gamma=frac(z\ B)\*, \*z\ B\ = z-0.5,\)

\(\alpha\), \(\beta\), and \(\gamma\) are stored in 9-bit fixed point format with 8 bits of fractional value (so 1.0 is exactly represented).

[Figure 37](#linear-filtering-of-1-d-texture-of-4-texels) illustrates linear filtering of a one-dimensional texture with *N=4*.

![_images/linear-filtering-of-1-d-texture-of-4-texels.png](_images/linear-filtering-of-1-d-texture-of-4-texels.png)


Figure 37 Linear Filtering Mode[](#linear-filtering-of-1-d-texture-of-4-texels "Permalink to this image")