# 12.2.2. Thread block cancellation constraints

### 12.2.2. Thread block cancellation constraints[](#thread-block-cancellation-constraints "Permalink to this headline")

The constraints are related to failed cancellation requests:

* Submitting another cancellation request after **observing** a
  previously failed request is Undefined Behavior.

  In the two code examples below, assuming the first cancellation request
  fails, only the first example exhibits undefined behavior.
  The second example is correct because there is no observation between the
  cancellation requests:

  **Invalid code:**

  ```
  // First request:
  ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);

  // First request query:
  [Synchronize bar0 code here.]
  bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
  assert(!success0); // Observed failure; second cacellation will be invalid.

  // Second request - next line is Undefined Behavior:
  ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);
  ```

  **Valid code:**

  ```
  // First request:
  ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);

  // Second request:
  ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);

  // First request query:
  [Synchronize bar0 code here.]
  bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
  assert(!success0); // Observed failure; second cacellation was valid.
  ```
* Retrieving the thread block index of a failed cancellation request
  is Undefined Behavior.
* Submitting a cancellation request from multiple threads is not recommended.
  It results in the cancellation of multiple thread blocks
  and requires careful handling,
  such as:

  + Each submitting thread must proivde a unique `__shared__` result
    pointer to avoid data races.
  + If the same barrier is used for synchronization, the arrival and
    transaction counts must be adjusted accordingly.