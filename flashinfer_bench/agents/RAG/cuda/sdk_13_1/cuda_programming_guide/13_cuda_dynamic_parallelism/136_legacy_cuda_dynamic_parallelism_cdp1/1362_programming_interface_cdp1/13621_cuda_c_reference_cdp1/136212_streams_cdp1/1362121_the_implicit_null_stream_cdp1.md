# 13.6.2.1.2.1. The Implicit (NULL) Stream (CDP1)

###### 13.6.2.1.2.1. The Implicit (NULL) Stream (CDP1)[](#the-implicit-null-stream-cdp1 "Permalink to this headline")

See [The Implicit (NULL) Stream](#the-implicit-null-stream), above, for CDP2 version of document.

Within a host program, the unnamed (NULL) stream has additional barrier synchronization semantics with other streams (see [Default Stream](#default-stream) for details). The device runtime offers a single implicit, unnamed stream shared between all threads in a block, but as all named streams must be created with the `cudaStreamNonBlocking` flag, work launched into the NULL stream will not insert an implicit dependency on pending work in any other streams (including NULL streams of other thread blocks).