# 13.6.1.2.1.2. Zero Copy Memory (CDP1)

###### 13.6.1.2.1.2. Zero Copy Memory (CDP1)[](#zero-copy-memory-cdp1 "Permalink to this headline")

See [Zero Copy Memory](#zero-copy-memory), above, for CDP2 version of document.

Zero-copy system memory has identical coherence and consistency guarantees to global memory, and follows the semantics detailed above. A kernel may not allocate or free zero-copy memory, but may use pointers to zero-copy passed in from the host program.