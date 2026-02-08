# 5.2.5. Packed Data Types

### 5.2.5. [Packed Data Types](https://docs.nvidia.com/cuda/parallel-thread-execution/#packed-data-types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#packed-data-types "Permalink to this headline")

Certain PTX instructions operate on two or more sets of inputs in parallel, and produce two or more
outputs. Such instructions can use the data stored in a packed format. PTX supports packing two or
four values of the same scalar data type into a single, larger value. The packed value is considered
as a value of a *packed data type*. In this section we describe the packed data types supported in PTX.