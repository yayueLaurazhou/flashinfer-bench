# __host__cudaError_t cudaMemcpyBatchAsync (const void **dsts, const void **srcs, const size_t *sizes, size_t count, cudaMemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs, cudaStream_t stream)

Performs a batch of memory copies asynchronously.

##### Parameters

**dsts**

  - Array of destination pointers.


CUDA Runtime API vRelease Version  |  172


Modules


**srcs**

  - Array of memcpy source pointers.
**sizes**

  - Array of sizes for memcpy operations.
**count**

  - Size of dsts, srcs and sizes arrays
**attrs**

  - Array of memcpy attributes.
**attrsIdxs**

  - Array of indices to specify which copies each entry in the attrs array applies to. The attributes
specified in attrs[k] will be applied to copies starting from attrsIdxs[k] through attrsIdxs[k+1] - 1.
Also attrs[numAttrs-1] will apply to copies starting from attrsIdxs[numAttrs-1] through count - 1.
**numAttrs**

  - Size of attrs and attrsIdxs arrays.
**stream**

##### Returns

cudaSuccess cudaErrorInvalidValue

##### Description

Performs a batch of memory copies. The batch as a whole executes in stream order but copies within
a batch are not guaranteed to execute in any specific order. This API only supports pointer-to-pointer
copies. For copies involving CUDA arrays, please see cudaMemcpy3DBatchAsync.

Performs memory copies from source buffers specified in srcs to destination buffers specified in
dsts. The size of each copy is specified in sizes. All three arrays must be of the same length as
specified by count. Since there are no ordering guarantees for copies within a batch, specifying any
dependent copies within a batch will result in undefined behavior.

Every copy in the batch has to be associated with a set of attributes specified in the attrs array. Each
entry in this array can apply to more than one copy. This can be done by specifying in the attrsIdxs
array, the index of the first copy that the corresponding entry in the attrs array applies to. Both
attrs and attrsIdxs must be of the same length as specified by numAttrs. For example, if
a batch has 10 copies listed in dst/src/sizes, the first 6 of which have one set of attributes and the
remaining 4 another, then numAttrs will be 2, attrsIdxs will be {0, 6} and attrs will contains
the two sets of attributes. Note that the first entry in attrsIdxs must always be 0. Also, each entry
must be greater than the previous entry and the last entry should be less than count. Furthermore,
numAttrs must be lesser than or equal to count.

The cudaMemcpyAttributes::srcAccessOrder indicates the source access ordering to
be observed for copies associated with the attribute. If the source access order is set to
cudaMemcpySrcAccessOrderStream, then the source will be accessed in stream order. If the source
access order is set to cudaMemcpySrcAccessOrderDuringApiCall then it indicates that access to
the source pointer can be out of stream order and all accesses must be complete before the API call
returns. This flag is suited for ephemeral sources (ex., stack variables) when it's known that no prior


CUDA Runtime API vRelease Version  |  173


Modules


operations in the stream can be accessing the memory and also that the lifetime of the memory is
limited to the scope that the source variable was declared in. Specifying this flag allows the driver to
optimize the copy and removes the need for the user to synchronize the stream after the API call. If
the source access order is set to cudaMemcpySrcAccessOrderAny then it indicates that access to the
source pointer can be out of stream order and the accesses can happen even after the API call returns.
This flag is suited for host pointers allocated outside CUDA (ex., via malloc) when it's known that
no prior operations in the stream can be accessing the memory. Specifying this flag allows the driver
to optimize the copy on certain platforms. Each memcpy operation in the batch must have a valid
cudaMemcpyAttributes corresponding to it including the appropriate srcAccessOrder setting, otherwise
the API will return cudaErrorInvalidValue.

The cudaMemcpyAttributes::srcLocHint and cudaMemcpyAttributes::dstLocHint allows
applications to specify hint locations for operands of a copy when the operand doesn't have a fixed
location. That is, these hints are only applicable for managed memory pointers on devices where
cudaDevAttrConcurrentManagedAccess is true or system-allocated pageable memory on devices
where cudaDevAttrPageableMemoryAccess is true. For other cases, these hints are ignored.

The cudaMemcpyAttributes::flags field can be used to specify certain flags for copies. Setting
the cudaMemcpyFlagPreferOverlapWithCompute flag indicates that the associated copies should
preferably overlap with any compute work. Note that this flag is a hint and can be ignored depending
on the platform and other parameters of the copy.