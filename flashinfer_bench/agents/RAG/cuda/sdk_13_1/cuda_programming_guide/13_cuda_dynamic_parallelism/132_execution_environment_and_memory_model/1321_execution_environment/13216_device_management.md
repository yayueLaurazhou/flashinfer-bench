# 13.2.1.6. Device Management

#### 13.2.1.6. Device Management[](#device-management "Permalink to this headline")

There is no multi-GPU support from the device runtime; the device runtime is only capable of operating on the device upon which it is currently executing. It is permitted, however, to query properties for any CUDA capable device in the system.