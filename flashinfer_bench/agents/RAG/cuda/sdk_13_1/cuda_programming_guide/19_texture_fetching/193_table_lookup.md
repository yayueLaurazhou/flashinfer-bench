# 19.3. Table Lookup

## 19.3. Table Lookup[](#table-lookup "Permalink to this headline")

A table lookup *TL(x)* where *x* spans the interval *[0,R]* can be implemented as *TL(x)=tex((N-1)/R)x+0.5)* in order to ensure that *TL(0)=T[0]* and *TL(R)=T[N-1]*.

[Figure 38](#table-lookup-1-d-table-lookup-using-linear-filtering) illustrates the use of texture filtering to implement a table lookup with *R=4* or *R=1* from a one-dimensional texture with *N=4*.

![_images/1-d-table-lookup-using-linear-filtering.png](_images/1-d-table-lookup-using-linear-filtering.png)


Figure 38 One-Dimensional Table Lookup Using Linear Filtering[](#table-lookup-1-d-table-lookup-using-linear-filtering "Permalink to this image")