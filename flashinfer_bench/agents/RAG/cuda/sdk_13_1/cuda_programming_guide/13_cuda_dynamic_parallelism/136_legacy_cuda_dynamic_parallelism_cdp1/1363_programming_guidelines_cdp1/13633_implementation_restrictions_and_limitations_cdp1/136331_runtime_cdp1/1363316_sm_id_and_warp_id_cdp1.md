# 13.6.3.3.1.6. SM Id and Warp Id (CDP1)

###### 13.6.3.3.1.6. SM Id and Warp Id (CDP1)[](#sm-id-and-warp-id-cdp1 "Permalink to this headline")

See [SM Id and Warp Id](#sm-id-and-warp-id), above, for CDP2 version of document.

Note that in PTX `%smid` and `%warpid` are defined as volatile values. The device runtime may reschedule thread blocks onto different SMs in order to more efficiently manage resources. As such, it is unsafe to rely upon `%smid` or `%warpid` remaining unchanged across the lifetime of a thread or thread block.