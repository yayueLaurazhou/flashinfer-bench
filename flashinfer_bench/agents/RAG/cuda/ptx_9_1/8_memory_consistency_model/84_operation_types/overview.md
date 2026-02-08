# 8.4. Operation types

## 8.4. [Operation types](https://docs.nvidia.com/cuda/parallel-thread-execution/#operation-types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#operation-types "Permalink to this headline")

For simplicity, the rest of the document refers to the following operation types, instead of
mentioning specific instructions that give rise to them.

Table 20 Operation Types[](https://docs.nvidia.com/cuda/parallel-thread-execution/#id678 "Permalink to this table")




| Operation Type | Instruction/Operation |
| --- | --- |
| atomic operation | `atom` or `red` instruction. |
| read operation | All variants of `ld` instruction and `atom` instruction (but not `red` instruction). |
| write operation | All variants of `st` instruction, and *atomic* operations if they result in a write. |
| memory operation | A *read* or *write* operation. |
| volatile operation | An instruction with `.volatile` qualifier. |
| acquire operation | A *memory* operation with `.acquire` or `.acq_rel` qualifier. |
| release operation | A *memory* operation with `.release` or `.acq_rel` qualifier. |
| mmio operation | An `ld` or `st` instruction with `.mmio` qualifier. |
| memory fence operation | A `membar`, `fence.sc` or `fence.acq_rel` instruction. |
| proxy fence operation | A `fence.proxy` or a `membar.proxy` instruction. |
| strong operation | A *memory fence* operation, or a *memory* operation with a `.relaxed`, `.acquire`, `.release`, `.acq_rel`, `.volatile`, or `.mmio` qualifier. |
| weak operation | An `ld` or `st` instruction with a `.weak` qualifier. |
| synchronizing operation | A `barrier` instruction, *fence* operation, *release* operation or *acquire* operation. |