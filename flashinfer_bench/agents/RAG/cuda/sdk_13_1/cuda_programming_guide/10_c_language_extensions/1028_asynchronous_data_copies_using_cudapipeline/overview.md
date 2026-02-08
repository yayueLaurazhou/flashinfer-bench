# 10.28. Asynchronous Data Copies using cuda::pipeline

## 10.28. Asynchronous Data Copies using `cuda::pipeline`[](#asynchronous-data-copies-using-cuda-pipeline "Permalink to this headline")

CUDA provides the `cuda::pipeline` synchronization object to manage and overlap asynchronous data movement with computation.

The API documentation for `cuda::pipeline` is provided in the [libcudacxx API](https://nvidia.github.io/libcudacxx). A pipeline object is a double-ended N stage queue with a *head* and a *tail*, and is used to process work in a first-in first-out (FIFO) order. The pipeline object has following member functions to manage the stages of the pipeline.

| Pipeline Class Member Function | Description |
| --- | --- |
| `producer_acquire` | Acquires an available stage in the pipeline internal queue. |
| `producer_commit` | Commits the asynchronous operations issued after the `producer_acquire` call on the currently acquired stage of the pipeline. |
| `consumer_wait` | Wait for completion of all asynchronous operations on the oldest stage of the pipeline. |
| `consumer_release` | Release the oldest stage of the pipeline to the pipeline object for reuse. The released stage can be then acquired by the producer. |