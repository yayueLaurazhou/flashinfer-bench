# 13.4.3.1.6. ECC Errors

##### 13.4.3.1.6. ECC Errors[](#ecc-errors "Permalink to this headline")

No notification of ECC errors is available to code within a CUDA kernel. ECC errors are reported at the host side once the entire launch tree has completed. Any ECC errors which arise during execution of a nested program will either generate an exception or continue execution (depending upon error and configuration).