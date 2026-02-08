# 5.2.5.3. Packed Fixed-Point Data Types

#### 5.2.5.3. [Packed Fixed-Point Data Types](https://docs.nvidia.com/cuda/parallel-thread-execution/#packed-fixed-point-data-types)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#packed-fixed-point-data-types "Permalink to this headline")

PTX supports `.s2f6x2` packed fixed-point data type consisting of two `.s2f6` packed
fixed-point values. A register variable containing `.s2f6x2` value must be declared with
`.b16` type. Packed fixed-point data type cannot be used as fundamental type and is only
supported as instruction type.