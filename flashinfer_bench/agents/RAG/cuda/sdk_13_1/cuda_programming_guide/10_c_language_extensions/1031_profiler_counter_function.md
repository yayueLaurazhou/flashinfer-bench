# 10.31. Profiler Counter Function

## 10.31. Profiler Counter Function[](#profiler-counter-function "Permalink to this headline")

Each multiprocessor has a set of sixteen hardware counters that an application can increment with a single instruction by calling the `__prof_trigger()` function.

```
void __prof_trigger(int counter);
```

increments by one per warp the per-multiprocessor hardware counter of index `counter`. Counters 8 to 15 are reserved and should not be used by applications.

The value of counters 0, 1, …, 7 can be obtained via `nvprof` by `nvprof --events prof_trigger_0x` where `x` is 0, 1, …, 7. All counters are reset before each kernel launch (note that when collecting counters, kernel launches are synchronous as mentioned in [Concurrent Execution between Host and Device](#concurrent-execution-host-device)).