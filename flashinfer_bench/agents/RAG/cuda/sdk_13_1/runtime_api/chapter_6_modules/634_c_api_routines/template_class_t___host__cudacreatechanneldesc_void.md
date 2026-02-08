# template < class T > __host__cudaCreateChannelDesc (void)

[C++ API] Returns a channel descriptor using the specified format

##### Returns

Channel descriptor with format f

##### Description

Returns a channel descriptor with format f and number of bits of each component x, y, z, and w. The
cudaChannelFormatDesc is defined as:


where cudaChannelFormatKind is one of cudaChannelFormatKindSigned,
cudaChannelFormatKindUnsigned, cudaChannelFormatKindFloat,
cudaChannelFormatKindSignedNormalized8X1, cudaChannelFormatKindSignedNormalized8X2,
cudaChannelFormatKindSignedNormalized8X4, cudaChannelFormatKindUnsignedNormalized8X1,
cudaChannelFormatKindUnsignedNormalized8X2, cudaChannelFormatKindUnsignedNormalized8X4,


CUDA Runtime API vRelease Version  |  462


Modules


cudaChannelFormatKindSignedNormalized16X1, cudaChannelFormatKindSignedNormalized16X2,
cudaChannelFormatKindSignedNormalized16X4, cudaChannelFormatKindUnsignedNormalized16X1,
cudaChannelFormatKindUnsignedNormalized16X2,
cudaChannelFormatKindUnsignedNormalized16X4,
cudaChannelFormatKindUnsignedNormalized1010102 or cudaChannelFormatKindNV12.

The format is specified by the template specialization.

The template function specializes for the following scalar types: char, signed char, unsigned char, short,
unsigned short, int, unsigned int, long, unsigned long, and float. The template function specializes
for the following vector types: char{1|2|4}, uchar{1|2|4}, short{1|2|4}, ushort{1|2|4}, int{1|2|4},
uint{1|2|4}, long{1|2|4}, ulong{1|2|4}, float{1|2|4}. The template function specializes for following
cudaChannelFormatKind enum values: cudaChannelFormatKind{Uns|S}ignedNormalized{8|16}X{1|2|
4}, cudaChannelFormatKindUnsignedNormalized1010102 and cudaChannelFormatKindNV12.

Invoking the function on a type without a specialization defaults to creating a channel format of kind
cudaChannelFormatKindNone


See also:

cudaCreateChannelDesc ( Low level), cudaGetChannelDesc,