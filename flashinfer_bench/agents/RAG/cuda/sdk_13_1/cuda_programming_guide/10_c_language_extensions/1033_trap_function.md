# 10.33. Trap function

## 10.33. Trap function[](#trap-function "Permalink to this headline")

A trap operation can be initiated by calling the `__trap()` function from any device thread.

```
void __trap();
```

The execution of the kernel is aborted and an interrupt is raised in the host program.