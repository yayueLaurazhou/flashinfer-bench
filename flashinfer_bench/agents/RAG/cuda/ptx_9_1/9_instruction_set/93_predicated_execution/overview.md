# 9.3. Predicated Execution

## 9.3. [Predicated Execution](https://docs.nvidia.com/cuda/parallel-thread-execution/#predicated-execution)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#predicated-execution "Permalink to this headline")

In PTX, predicate registers are virtual and have `.pred` as the type specifier. So, predicate
registers can be declared as

```
.reg .pred p, q, r;
```

Copy to clipboard

All instructions have an optional *guard predicate* which controls conditional execution of the
instruction. The syntax to specify conditional execution is to prefix an instruction with `@{!}p`,
where `p` is a predicate variable, optionally negated. Instructions without a guard predicate are
executed unconditionally.

Predicates are most commonly set as the result of a comparison performed by the `setp`
instruction.

As an example, consider the high-level code

```
if (i < n)
    j = j + 1;
```

Copy to clipboard

This can be written in PTX as

```
      setp.lt.s32  p, i, n;    // p = (i < n)
@p    add.s32      j, j, 1;    // if i < n, add 1 to j
```

Copy to clipboard

To get a conditional branch or conditional function call, use a predicate to control the execution
of the branch or call instructions. To implement the above example as a true conditional branch, the
following PTX instruction sequence might be used:

```
      setp.lt.s32  p, i, n;    // compare i to n
@!p   bra  L1;                 // if False, branch over
      add.s32      j, j, 1;
L1:     ...
```

Copy to clipboard