# 9.7.18.1. Scalar Video Instructions

#### 9.7.18.1. [Scalar Video Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#scalar-video-instructions)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#scalar-video-instructions "Permalink to this headline")

All scalar video instructions operate on 32-bit register operands. The scalar video instructions
are:

* `vadd`
* `vsub`
* `vabsdiff`
* `vmin`
* `vmax`
* `vshl`
* `vshr`
* `vmad`
* `vset`

The scalar video instructions execute the following stages:

1. Extract and sign- or zero-extend byte, half-word, or word values from its source operands, to
   produce signed 33-bit input values.
2. Perform a scalar arithmetic operation to produce a signed 34-bit result.
3. Optionally clamp the result to the range of the destination type.
4. Optionally perform one of the following:

   * apply a second operation to the intermediate result and a third operand, or
   * truncate the intermediate result to a byte or half-word value and merge into a specified
     position in the third operand to produce the final result.

The general format of scalar video instructions is as follows:

```
// 32-bit scalar operation, with optional secondary operation
vop.dtype.atype.btype{.sat}        d, a{.asel}, b{.bsel};
vop.dtype.atype.btype{.sat}.secop  d, a{.asel}, b{.bsel}, c;

// 32-bit scalar operation, with optional data merge
vop.dtype.atype.btype{.sat}   d.dsel, a{.asel}, b{.bsel}, c;


.dtype = .atype = .btype = { .u32, .s32 };
.dsel  = .asel  = .bsel  = { .b0, .b1, .b2, .b3, .h0, .h1 };
.secop = { .add, .min, .max };
```

Copy to clipboard

The source and destination operands are all 32-bit registers. The type of each operand (`.u32` or
`.s32`) is specified in the instruction type; all combinations of `dtype`, `atype`, and
`btype` are valid. Using the `atype/btype` and `asel/bsel` specifiers, the input values are
extracted and sign- or zero-extended internally to `.s33` values. The primary operation is then
performed to produce an `.s34` intermediate result. The sign of the intermediate result depends on
dtype.

The intermediate result is optionally clamped to the range of the destination type (signed or
unsigned), taking into account the subword destination size in the case of optional data merging.

```
.s33 optSaturate( .s34 tmp, Bool sat, Bool sign, Modifier dsel ) {
    if ( !sat )  return tmp;

    switch ( dsel ) {
        case .b0, .b1, .b2, .b3:
            if ( sign )  return CLAMP( tmp, S8_MAX, S8_MIN );
            else         return CLAMP( tmp, U8_MAX, U8_MIN );
        case .h0, .h1:
            if ( sign )  return CLAMP( tmp, S16_MAX, S16_MIN );
            else         return CLAMP( tmp, U16_MAX, U16_MIN );
        default:
            if ( sign )  return CLAMP( tmp, S32_MAX, S32_MIN );
            else         return CLAMP( tmp, U32_MAX, U32_MIN );
    }
}
```

Copy to clipboard

This intermediate result is then optionally combined with the third source operand using a secondary
arithmetic operation or subword data merge, as shown in the following pseudocode. The sign of the
third operand is based on `dtype`.

```
.s33 optSecOp(Modifier secop, .s33 tmp, .s33 c) {
    switch ( secop ) {
        .add:     return tmp + c;
        .min:     return MIN(tmp, c);
        .max      return MAX(tmp, c);
        default:  return tmp;
    }
}
```

Copy to clipboard

```
.s33 optMerge( Modifier dsel, .s33 tmp, .s33 c ) {
    switch ( dsel ) {
        case .h0:  return ((tmp & 0xffff)        | (0xffff0000 & c);
        case .h1:  return ((tmp & 0xffff) << 16) | (0x0000ffff & c);
        case .b0:  return ((tmp & 0xff)          | (0xffffff00 & c);
        case .b1:  return ((tmp & 0xff) <<  8)   | (0xffff00ff & c);
        case .b2:  return ((tmp & 0xff) << 16)   | (0xff00ffff & c);
        case .b3:  return ((tmp & 0xff) << 24)   | (0x00ffffff & c);
        default:   return tmp;
    }
}
```

Copy to clipboard

The lower 32-bits are then written to the destination operand.