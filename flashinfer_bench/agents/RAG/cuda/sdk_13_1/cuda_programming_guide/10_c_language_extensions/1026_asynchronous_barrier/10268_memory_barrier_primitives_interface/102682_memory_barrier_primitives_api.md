# 10.26.8.2. Memory Barrier Primitives API

#### 10.26.8.2. Memory Barrier Primitives API[](#memory-barrier-primitives-api "Permalink to this headline")

```
uint32_t __mbarrier_maximum_count();
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);
```

* `bar` must be a pointer to `__shared__` memory.
* `expected_count <= __mbarrier_maximum_count()`
* Initialize `*bar` expected arrival count for the current and next phase to `expected_count`.

```
void __mbarrier_inval(__mbarrier_t* bar);
```

* `bar` must be a pointer to the mbarrier object residing in shared memory.
* Invalidation of `*bar` is required before the corresponding shared memory can be repurposed.

```
__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);
```

* Initialization of `*bar` must happen before this call.
* Pending count must not be zero.
* Atomically decrement the pending count for the current phase of the barrier.
* Return an arrival token associated with the barrier state immediately prior to the decrement.

```
__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);
```

* Initialization of `*bar` must happen before this call.
* Pending count must not be zero.
* Atomically decrement the pending count for the current phase and expected count for the next phase of the barrier.
* Return an arrival token associated with the barrier state immediately prior to the decrement.

```
bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);
```

* `token` must be associated with the immediately preceding phase or current phase of `*this`.
* Returns `true` if `token` is associated with the immediately preceding phase of `*bar`, otherwise returns `false`.

```
//Note: This API has been deprecated in CUDA 11.1
uint32_t __mbarrier_pending_count(__mbarrier_token_t token);
```