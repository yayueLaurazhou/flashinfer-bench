# 13.3.1.7.1. Launch Setup APIs

##### 13.3.1.7.1. Launch Setup APIs[](#launch-setup-apis "Permalink to this headline")

Kernel launch is a system-level mechanism exposed through the device runtime library, and as such is available directly from PTX via the underlying `cudaGetParameterBuffer()` and `cudaLaunchDevice()` APIs. It is permitted for a CUDA application to call these APIs itself, with the same requirements as for PTX. In both cases, the user is then responsible for correctly populating all necessary data structures in the correct format according to specification. Backwards compatibility is guaranteed in these data structures.

As with host-side launch, the device-side operator `<<<>>>` maps to underlying kernel launch APIs. This is so that users targeting PTX will be able to enact a launch, and so that the compiler front-end can translate `<<<>>>` into these calls.

Table 13 New Device-only Launch Implementation Functions[](#id471 "Permalink to this table")




| Runtime API Launch Functions | Description of Difference From Host Runtime Behaviour (behavior is identical if no description) |
| --- | --- |
| `cudaGetParameterBuffer` | Generated automatically from `<<<>>>`. Note different API to host equivalent. |
| `cudaLaunchDevice` | Generated automatically from `<<<>>>`. Note different API to host equivalent. |

The APIs for these launch functions are different to those of the CUDA Runtime API, and are defined as follows:

```
extern   device   cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                        void *params, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize = 0,
                                        cudaStream_t stream = 0);
```