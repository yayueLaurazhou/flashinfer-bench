# Chapter 2. API synchronization behavior

The API provides memcpy/memset functions in both synchronous and asynchronous forms, the
latter having an "Async" suffix. This is a misnomer as each function may exhibit synchronous or
asynchronous behavior depending on the arguments passed to the function. The synchronous forms of
these APIs issue these copies through the default stream.

Any CUDA API call may block or synchronize for various reasons such as contention for or
unavailability of internal resources. Such behavior is subject to change and undocumented behavior
should not be relied upon.