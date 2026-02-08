# 9.7.17. Stack Manipulation Instructions

### 9.7.17. [Stack Manipulation Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#stack-manipulation-instructions)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#stack-manipulation-instructions "Permalink to this headline")

The stack manipulation instructions can be used to dynamically allocate and deallocate memory on the
stack frame of the current function.

The stack manipulation instrucitons are:

* `stacksave`
* `stackrestore`
* `alloca`