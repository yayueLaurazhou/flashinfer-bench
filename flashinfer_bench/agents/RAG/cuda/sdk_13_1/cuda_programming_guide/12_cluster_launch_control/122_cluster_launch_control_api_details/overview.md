# 12.2. Cluster Launch Control API Details

## 12.2. Cluster Launch Control API Details[](#cluster-launch-control-api-details "Permalink to this headline")

Cancelling a thread block via the Cluster Launch Control API is done
asynchronously and synchronized using a memory barrier,
following a programming pattern similar
to [asynchronous data copies](#memcpy-async-barrier).

The API, currently available through
[libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx.html),
provides a request instruction that writes the encoded cancellation result
into a `__shared__` variable, along with instructions to decode the result
into a *Success*/*Fail* flag and the index of the cancelled thread block
in case of *Success*.