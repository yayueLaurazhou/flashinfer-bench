# 11.4.2.1.2. Single Thread Group

##### 11.4.2.1.2. Single Thread Group[](#single-thread-group "Permalink to this headline")

Group representing the current thread can be obtained from `this_thread` function:

```
thread_block_tile<1> this_thread();
```

The following `memcpy_async` API uses a `thread_group`, to copy an int element from source to destination:

```
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

cooperative_groups::memcpy_async(cooperative_groups::this_thread(), dest, src, sizeof(int));
```

More detailed examples of using `this_thread` to perform asynchronous copies can be found in the [Single-Stage Asynchronous Data Copies using cuda::pipeline](#with-memcpy-async-pipeline-pattern-single) and [Multi-Stage Asynchronous Data Copies using cuda::pipeline](#with-memcpy-async-pipeline-pattern-multi) sections.