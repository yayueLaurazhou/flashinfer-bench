# 13.6.3.3.1.7. ECC Errors (CDP1)

###### 13.6.3.3.1.7. ECC Errors (CDP1)[](#ecc-errors-cdp1 "Permalink to this headline")

See [ECC Errors](#ecc-errors), above, for CDP2 version of document.

No notification of ECC errors is available to code within a CUDA kernel. ECC errors are reported at the host side once the entire launch tree has completed. Any ECC errors which arise during execution of a nested program will either generate an exception or continue execution (depending upon error and configuration).

7([1](#id232),[2](#id281),[3](#id326))
:   Dynamically created texture and surface objects are an addition to the CUDA memory model introduced with CUDA 5.0. Please see the *CUDA Programming Guide* for details.