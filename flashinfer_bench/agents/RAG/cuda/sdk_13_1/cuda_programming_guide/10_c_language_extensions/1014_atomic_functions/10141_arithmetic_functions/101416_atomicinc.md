# 10.14.1.6. atomicInc()

#### 10.14.1.6. atomicInc()[](#atomicinc "Permalink to this headline")

```
unsigned int atomicInc(unsigned int* address,
                       unsigned int val);
```

reads the 32-bit word `old` located at the address `address` in global or shared memory, computes `((old >= val) ? 0 : (old+1))`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.