# 13.4.3.1.5. SM Id and Warp Id

##### 13.4.3.1.5. SM Id and Warp Id[](#sm-id-and-warp-id "Permalink to this headline")

Note that in PTX `%smid` and `%warpid` are defined as volatile values. The device runtime may reschedule thread blocks onto different SMs in order to more efficiently manage resources. As such, it is unsafe to rely upon `%smid` or `%warpid` remaining unchanged across the lifetime of a thread or thread block.