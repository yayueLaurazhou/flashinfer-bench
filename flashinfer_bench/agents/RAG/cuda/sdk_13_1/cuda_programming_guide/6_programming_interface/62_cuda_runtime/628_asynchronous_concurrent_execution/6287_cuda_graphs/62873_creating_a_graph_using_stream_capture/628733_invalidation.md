# 6.2.8.7.3.3. Invalidation

###### 6.2.8.7.3.3. Invalidation[](#invalidation "Permalink to this headline")

When an invalid operation is attempted during stream capture, any associated capture graphs are *invalidated*. When a capture graph is invalidated, further use of any streams which are being captured or captured events associated with the graph is invalid and will return an error, until stream capture is ended with `cudaStreamEndCapture()`. This call will take the associated streams out of capture mode, but will also return an error value and a NULL graph.