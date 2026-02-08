# 13.6.3.2.2. Dynamic-parallelism-enabled Kernel Overhead (CDP1)

##### 13.6.3.2.2. Dynamic-parallelism-enabled Kernel Overhead (CDP1)[](#dynamic-parallelism-enabled-kernel-overhead-cdp1 "Permalink to this headline")

See [Dynamic-parallelism-enabled Kernel Overhead](#dynamic-parallelism-enabled-kernel-overhead), above, for CDP2 version of document.

System software which is active when controlling dynamic launches may impose an overhead on any kernel which is running at the time, whether or not it invokes kernel launches of its own. This overhead arises from the device runtime’s execution tracking and management software and may result in decreased performance for example, library calls when made from the device compared to from the host side. This overhead is, in general, incurred for applications that link against the device runtime library.