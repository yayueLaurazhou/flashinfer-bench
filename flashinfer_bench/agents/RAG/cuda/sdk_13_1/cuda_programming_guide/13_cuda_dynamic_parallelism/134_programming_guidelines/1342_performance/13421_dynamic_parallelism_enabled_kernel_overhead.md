# 13.4.2.1. Dynamic-parallelism-enabled Kernel Overhead

#### 13.4.2.1. Dynamic-parallelism-enabled Kernel Overhead[](#dynamic-parallelism-enabled-kernel-overhead "Permalink to this headline")

System software which is active when controlling dynamic launches may impose an overhead on any kernel which is running at the time, whether or not it invokes kernel launches of its own. This overhead arises from the device runtime’s execution tracking and management software and may result in decreased performance. This overhead is, in general, incurred for applications that link against the device runtime library.