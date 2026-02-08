# 10.18.5. __builtin_unreachable()

### 10.18.5. \_\_builtin\_unreachable()[](#builtin-unreachable "Permalink to this headline")

```
void __builtin_unreachable(void)
```

Indicates to the compiler that control flow never reaches the point where this function is being called from. The program has undefined behavior if the control flow does actually reach this point at run time.

Example:

```
// indicates to the compiler that the default case label is never reached.
switch (in) {
case 1: return 4;
case 2: return 10;
default: __builtin_unreachable();
}
```