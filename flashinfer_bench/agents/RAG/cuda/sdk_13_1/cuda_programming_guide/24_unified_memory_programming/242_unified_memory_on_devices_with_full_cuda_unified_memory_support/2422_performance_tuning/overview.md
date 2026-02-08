# 24.2.2. Performance Tuning

### 24.2.2. Performance Tuning[](#performance-tuning "Permalink to this headline")

In order to achieve good performance with Unified Memory, it is important to:

* Understand how paging works on your system, and how to avoid unnecessary page faults.
* Understand the various mechanisms allowing you to keep data local to the accessing processor.
* Consider tuning your application for the granularity of memory transfers of your system.

As general advice, [Performance Hints](#um-perf-hints)
might provide improved performance, but using them incorrectly might degrade performance
compared to the default behavior.
Also note that any hint has a performance cost associated with it on the host,
thus useful hints must at the very least improve performance enough to overcome this cost.