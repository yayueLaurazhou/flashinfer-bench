# 13.3.1.4. Synchronization

#### 13.3.1.4. Synchronization[](#synchronization-programming-interface "Permalink to this headline")

It is up to the program to perform sufficient inter-thread synchronization, for example via a CUDA Event, if the calling thread is intended to synchronize with child grids invoked from other threads.

As it is not possible to explicitly synchronize child work from a parent thread, there is no way to guarantee that changes occurring in child grids are visible to threads within the parent grid.