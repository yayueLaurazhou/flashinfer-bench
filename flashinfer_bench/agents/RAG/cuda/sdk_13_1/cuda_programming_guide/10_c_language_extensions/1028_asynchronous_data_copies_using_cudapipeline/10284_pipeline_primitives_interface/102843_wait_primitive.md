# 10.28.4.3. Wait Primitive

#### 10.28.4.3. Wait Primitive[](#wait-primitive "Permalink to this headline")

```
void __pipeline_wait_prior(size_t N);
```

* Let `{0, 1, 2, ..., L}` be the sequence of indices associated with invocations of `__pipeline_commit()` by a given thread.
* Wait for completion of batches *at least* up to and including `L-N`.