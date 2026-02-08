# 8.11.1. Reductions do not form Acquire Patterns

### 8.11.1. [Reductions do not form Acquire Patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/#red-read)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#red-read "Permalink to this headline")

Atomic reduction operations like `red` do not form acquire patterns with acquire fences.

**Litmus Test: Message Passing with a Red Instruction**

|  |  |
| --- | --- |
| ``` .global .u32 x = 0; .global .u32 flag = 0; ```  Copy to clipboard | |
| T1 | T2 |
| ``` W1: st.u32 [x], 42; W2: st.release.gpu.u32 [flag], 1; ```  Copy to clipboard | ``` RMW1: red.sys.global.add.u32 [flag], 1; F2: fence.acquire.gpu; R2: ld.weak.u32 %r1, [x]; ```  Copy to clipboard |
| ``` %r1 == 0 AND flag == 2 ```  Copy to clipboard | |

The litmus test known as “MP” (Message Passing) demonstrates the consequence
of reductions being excluded from acquire patterns.
It is possible to observe the outcome where `R2` reads the value `0`
from `x` and `flag` has the final value of `2`.
This outcome is possible since the release pattern in `T1` does not synchronize
with any acquire pattern in `T2`.
Using the `atom` instruction instead of `red` forbids this outcome.