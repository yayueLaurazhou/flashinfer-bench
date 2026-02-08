# 10.2.5. __managed__

### 10.2.5. \_\_managed\_\_[](#managed "Permalink to this headline")

The `__managed__` memory space specifier, optionally used together with `__device__`, declares a variable that:

* Can be referenced from both device and host code, for example, its address can be taken or it can be read or written directly from a device or host function.
* Has the lifetime of an application.

See [\_\_managed\_\_ Memory Space Specifier](#managed-specifier) for more details.