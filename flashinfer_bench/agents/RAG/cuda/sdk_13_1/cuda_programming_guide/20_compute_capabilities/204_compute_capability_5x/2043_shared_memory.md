# 20.4.3. Shared Memory

### 20.4.3. Shared Memory[](#shared-memory-5-x "Permalink to this headline")

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle.

A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word (even though the two addresses fall in the same bank). In that case, for read accesses, the word is broadcast to the requesting threads and for write accesses, each address is written by only one of the threads (which thread performs the write is undefined).

[Figure 39](#shared-memory-5-x-examples-of-strided-shared-memory-accesses) shows some examples of strided access.

[Figure 40](#shared-memory-5-x-examples-of-irregular-shared-memory-accesses) shows some examples of memory read accesses that involve the broadcast mechanism.

![Strided Shared Memory Accesses in 32 bit bank size mode.](_images/examples-of-strided-shared-memory-accesses.png)


Figure 39 Strided Shared Memory Accesses in 32 bit bank size mode.[](#shared-memory-5-x-examples-of-strided-shared-memory-accesses "Permalink to this image")

Left
:   Linear addressing with a stride of one 32-bit word (no bank conflict).

Middle
:   Linear addressing with a stride of two 32-bit words (two-way bank conflict).

Right
:   Linear addressing with a stride of three 32-bit words (no bank conflict).


![Irregular Shared Memory Accesses.](_images/examples-of-irregular-shared-memory-accesses.png)


Figure 40 Irregular Shared Memory Accesses.[](#shared-memory-5-x-examples-of-irregular-shared-memory-accesses "Permalink to this image")

Left
:   Conflict-free access via random permutation.

Middle
:   Conflict-free access since threads 3, 4, 6, 7, and 9 access the same word within bank 5.

Right
:   Conflict-free broadcast access (threads access the same word within a bank).