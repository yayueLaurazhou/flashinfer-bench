# 4.3.1. Directive Statements

### 4.3.1. [Directive Statements](https://docs.nvidia.com/cuda/parallel-thread-execution/#directive-statements)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#directive-statements "Permalink to this headline")

Directive keywords begin with a dot, so no conflict is possible with user-defined identifiers. The
directives in PTX are listed in [Table 1](https://docs.nvidia.com/cuda/parallel-thread-execution/#directive-statements-ptx-directives) and
described in [State Spaces, Types, and Variables](https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces-types-and-variables)
and [Directives](https://docs.nvidia.com/cuda/parallel-thread-execution/#directives).

Table 1 PTX Directives[](https://docs.nvidia.com/cuda/parallel-thread-execution/#directive-statements-ptx-directives "Permalink to this table")






|  |  |  |  |
| --- | --- | --- | --- |
| `.address_size` | `.explicitcluster` | `.maxnreg` | `.section` |
| `.alias` | `.extern` | `.maxntid` | `.shared` |
| `.align` | `.file` | `.minnctapersm` | `.sreg` |
| `.branchtargets` | `.func` | `.noreturn` | `.target` |
| `.callprototype` | `.global` | `.param` | `.tex` |
| `.calltargets` | `.loc` | `.pragma` | `.version` |
| `.common` | `.local` | `.reg` | `.visible` |
| `.const` | `.maxclusterrank` | `.reqnctapercluster` | `.weak` |
| `.entry` | `.maxnctapersm` | `.reqntid` |  |