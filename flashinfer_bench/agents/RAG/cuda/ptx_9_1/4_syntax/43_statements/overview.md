# 4.3. Statements

## 4.3. [Statements](https://docs.nvidia.com/cuda/parallel-thread-execution/#statements)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#statements "Permalink to this headline")

A PTX statement is either a directive or an instruction. Statements begin with an optional label and
end with a semicolon.

Examples

```
        .reg     .b32 r1, r2;
        .global  .f32  array[N];

start:  mov.b32   r1, %tid.x;
        shl.b32   r1, r1, 2;          // shift thread id by 2 bits
        ld.global.b32 r2, array[r1];  // thread[tid] gets array[tid]
        add.f32   r2, r2, 0.5;        // add 1/2
```

Copy to clipboard