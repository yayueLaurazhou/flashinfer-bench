# 18.5.3.1. Device Memory Space Specifiers

#### 18.5.3.1. Device Memory Space Specifiers[](#device-memory-space-specifiers "Permalink to this headline")

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on:

* `class`, `struct`, and `union` data members,
* formal parameters,
* non-extern variable declarations within a function that executes on the host.

The `__device__`, `__constant__` and `__managed__` memory space specifiers are not allowed on variable declarations that are neither extern nor static within a function that executes on the device.

A `__device__`, `__constant__`, `__managed__` or `__shared__` variable definition cannot have a class type with a non-empty constructor or a non-empty destructor. A constructor for a class type is considered empty at a point in the translation unit, if it is either a trivial constructor or it satisfies all of the following conditions:

* The constructor function has been defined.
* The constructor function has no parameters, the initializer list is empty and the function body is an empty compound statement.
* Its class has no virtual functions, no virtual base classes and no non-static data member initializers.
* The default constructors of all base classes of its class can be considered empty.
* For all the nonstatic data members of its class that are of class type (or array thereof), the default constructors can be considered empty.

A destructor for a class is considered empty at a point in the translation unit, if it is either a trivial destructor or it satisfies all of the following conditions:

* The destructor function has been defined.
* The destructor function body is an empty compound statement.
* Its class has no virtual functions and no virtual base classes.
* The destructors of all base classes of its class can be considered empty.
* For all the nonstatic data members of its class that are of class type (or array thereof), the destructor can be considered empty.

When compiling in the whole program compilation mode (see the nvcc user manual for a description of this mode), `__device__`, `__shared__`, `__managed__` and `__constant__` variables cannot be defined as external using the `extern` keyword. The only exception is for dynamically allocated `__shared__` variables as described in [\_\_shared\_\_](#shared).

When compiling in the separate compilation mode (see the nvcc user manual for a description of this mode), `__device__`, `__shared__`, `__managed__` and `__constant__` variables can be defined as external using the `extern` keyword. `nvlink` will generate an error when it cannot find a definition for an external variable (unless it is a dynamically allocated `__shared__` variable).