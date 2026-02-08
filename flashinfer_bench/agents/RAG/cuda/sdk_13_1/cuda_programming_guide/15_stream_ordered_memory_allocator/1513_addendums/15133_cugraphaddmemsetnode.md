# 15.13.3. cuGraphAddMemsetNode

### 15.13.3. cuGraphAddMemsetNode[](#cugraphaddmemsetnode "Permalink to this headline")

`cuGraphAddMemsetNode` does not work with memory allocated via the stream ordered allocator. However, memsets of the allocations can be stream captured.