# 11.6. Group Collectives

## 11.6. Group Collectives[](#group-collectives "Permalink to this headline")

Cooperative Groups library provides a set of collective operations that can be performed by a group of threads.
These operations require participation of all threads in the specified group in order to complete the operation.
All threads in the group need to pass the same values for corresponding arguments to each collective call, unless
different values are explicitly allowed in the argument description. Otherwise the behavior of the call is undefined.