# 10.14.1.7. atomicDec()

#### 10.14.1.7. atomicDec()[](#atomicdec "Permalink to this headline")

```
unsigned int atomicDec(unsigned int* address,
                       unsigned int val);
```

reads the 32-bit word `old` located at the address `address` in global or shared memory, computes `(((old == 0) || (old > val)) ? val : (old-1)` ), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.