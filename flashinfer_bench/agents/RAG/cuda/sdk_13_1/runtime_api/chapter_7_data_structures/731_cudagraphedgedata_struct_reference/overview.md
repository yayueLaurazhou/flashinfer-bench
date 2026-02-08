# 7.31. cudaGraphEdgeData Struct Reference

Optional annotation for edges in a CUDA graph. Note, all edges implicitly have annotations and
default to a zero-initialized value if not specified. A zero-initialized struct indicates a standard full
serialization of two nodes with memory visibility.