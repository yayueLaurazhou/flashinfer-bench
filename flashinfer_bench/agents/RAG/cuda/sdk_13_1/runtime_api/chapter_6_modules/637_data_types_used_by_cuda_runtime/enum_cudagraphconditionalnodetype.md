# enum cudaGraphConditionalNodeType

CUDA conditional node types

##### Values

**cudaGraphCondTypeIf = 0**
Conditional 'if/else' Node. Body[0] executed if condition is non-zero. If size == 2, an optional
ELSE graph is created and this is executed if the condition is zero.
**cudaGraphCondTypeWhile = 1**
Conditional 'while' Node. Body executed repeatedly while condition value is non-zero.
**cudaGraphCondTypeSwitch = 2**
Conditional 'switch' Node. Body[n] is executed once, where 'n' is the value of the condition. If the
condition does not match a body index, no body is launched.