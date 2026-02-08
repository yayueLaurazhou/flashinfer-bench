# 18.5.5.1. Assignment Operator

#### 18.5.5.1. Assignment Operator[](#assignment-operator "Permalink to this headline")

`__constant__` variables can only be assigned from the host code through runtime functions ([Device Memory](#device-memory)); they cannot be assigned from the device code.

`__shared__` variables cannot have an initialization as part of their declaration.

It is not allowed to assign values to any of the built-in variables defined in [Built-in Variables](#built-in-variables).