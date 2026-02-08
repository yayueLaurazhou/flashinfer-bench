# 13.6.2.2.1. Kernel Launch APIs (CDP1)

##### 13.6.2.2.1. Kernel Launch APIs (CDP1)[](#kernel-launch-apis-cdp1 "Permalink to this headline")

See [Kernel Launch APIs](#id237), above, for CDP2 version of document.

Device-side kernel launches can be implemented using the following two APIs accessible from PTX: `cudaLaunchDevice()` and `cudaGetParameterBuffer()`. `cudaLaunchDevice()` launches the specified kernel with the parameter buffer that is obtained by calling `cudaGetParameterBuffer()` and filled with the parameters to the launched kernel. The parameter buffer can be NULL, i.e., no need to invoke `cudaGetParameterBuffer()`, if the launched kernel does not take any parameters.