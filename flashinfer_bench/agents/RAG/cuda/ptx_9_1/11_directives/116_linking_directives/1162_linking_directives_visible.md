# 11.6.2. Linking Directives: .visible

### 11.6.2. [Linking Directives: `.visible`](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-visible)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-visible "Permalink to this headline")

`.visible`

Visible (externally) symbol declaration.

Syntax

```
.visible identifier
```

Copy to clipboard

Description

Declares identifier to be globally visible. Unlike C, where identifiers are globally visible unless
declared static, PTX identifiers are visible only within the current module unless declared
`.visible` outside the current.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
.visible .global .b32 foo;  // foo will be externally visible
```

Copy to clipboard