# 24.1.2.8. Performance Hints

#### 24.1.2.8. Performance Hints[](#performance-hints "Permalink to this headline")

The following sections describes the available unified memory performance hints,
which may be used on all Unified Memory, for example, CUDA Managed memory or,
on [systems with full CUDA Unified Memory support](#um-requirements),
also all System-Allocated Memory.
These APIs are hints, that is, they do not impact the semantics of applications, only their peformance.
That is, they can be added or removed anywhere on any application without impacting its results.

CUDA Unified Memory may not always have all the information necessary to make
the best performance decisions related to unified memory.
These performance hints enable the application to provide CUDA with more information.

Note that applications should only use these hints if they improve their performance.