# 13.2.2.1.2. Zero Copy Memory

##### 13.2.2.1.2. Zero Copy Memory[](#zero-copy-memory "Permalink to this headline")

Zero-copy system memory has identical coherence and consistency guarantees to global memory, and follows the semantics detailed above. A kernel may not allocate or free zero-copy memory, but may use pointers to zero-copy passed in from the host program.