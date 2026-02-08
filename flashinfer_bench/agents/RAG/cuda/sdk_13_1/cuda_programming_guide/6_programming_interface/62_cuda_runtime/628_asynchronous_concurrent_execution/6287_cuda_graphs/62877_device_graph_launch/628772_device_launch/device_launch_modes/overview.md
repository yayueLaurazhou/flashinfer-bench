# device-launch-modes

###### 6.2.8.7.7.2.1. Device Launch Modes[](#device-launch-modes "Permalink to this headline")

Unlike host launch, device graphs cannot be launched into regular CUDA streams, and can only be launched into distinct named streams, which each denote a specific launch mode:

Table 5 Device-only Graph Launch Streams[](#id456 "Permalink to this table")




| Stream | Launch Mode |
| --- | --- |
| `cudaStreamGraphFireAndForget` | Fire and forget launch |
| `cudaStreamGraphTailLaunch` | Tail launch |
| `cudaStreamGraphFireAndForgetAsSibling` | Sibling launch |