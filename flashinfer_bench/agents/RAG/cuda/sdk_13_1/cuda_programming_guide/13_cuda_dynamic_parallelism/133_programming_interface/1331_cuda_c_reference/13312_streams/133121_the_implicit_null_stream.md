# 13.3.1.2.1. The Implicit (NULL) Stream

##### 13.3.1.2.1. The Implicit (NULL) Stream[](#the-implicit-null-stream "Permalink to this headline")

Within a host program, the unnamed (NULL) stream has additional barrier synchronization semantics with other streams (see [Default Stream](#default-stream) for details). The device runtime offers a single implicit, unnamed stream shared between all threads in a thread block, but as all named streams must be created with the `cudaStreamNonBlocking` flag, work launched into the NULL stream will not insert an implicit dependency on pending work in any other streams (including NULL streams of other thread blocks).