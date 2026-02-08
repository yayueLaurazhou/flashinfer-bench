# 11.6.1. Linking Directives: .extern

### 11.6.1. [Linking Directives: `.extern`](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-extern)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-extern "Permalink to this headline")

`.extern`

External symbol declaration.

Syntax

```
.extern identifier
```

Copy to clipboard

Description

Declares identifier to be defined external to the current module. The module defining such
identifier must define it as `.weak` or `.visible` only once in a single object file. Extern
declaration of symbol may appear multiple times and references to that get resolved against the
single definition of that symbol.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
.extern .global .b32 foo;  // foo is defined in another module
```

Copy to clipboard