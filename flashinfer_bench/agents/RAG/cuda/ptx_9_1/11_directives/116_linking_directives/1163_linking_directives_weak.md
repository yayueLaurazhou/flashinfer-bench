# 11.6.3. Linking Directives: .weak

### 11.6.3. [Linking Directives: `.weak`](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-weak)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#linking-directives-weak "Permalink to this headline")

`.weak`

Visible (externally) symbol declaration.

Syntax

```
.weak identifier
```

Copy to clipboard

Description

Declares identifier to be globally visible but *weak*. Weak symbols are similar to globally visible
symbols, except during linking, weak symbols are only chosen after globally visible symbols during
symbol resolution. Unlike globally visible symbols, multiple object files may declare the same weak
symbol, and references to a symbol get resolved against a weak symbol only if no global symbols have
the same name.

PTX ISA Notes

Introduced in PTX ISA version 3.1.

Target ISA Notes

Supported on all target architectures.

Examples

```
.weak .func (.reg .b32 val) foo;  // foo will be externally visible
```

Copy to clipboard