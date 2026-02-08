# 10.28.4.4. Arrive On Barrier Primitive

#### 10.28.4.4. Arrive On Barrier Primitive[](#arrive-on-barrier-primitive "Permalink to this headline")

```
void __pipeline_arrive_on(__mbarrier_t* bar);
```

* `bar` points to a barrier in shared memory.
* Increments the barrier arrival count by one, when all memcpy\_async operations sequenced before this call have completed, the arrival count is decremented by one and hence the net effect on the arrival count is zero. It is user’s responsibility to make sure that the increment on the arrival count does not exceed `__mbarrier_maximum_count()`.