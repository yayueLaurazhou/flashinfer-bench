# 23.3. Output

## 23.3. Output[](#output "Permalink to this headline")

Logs are output in the following format:

```
[Time][TID][Source][Severity][API Entry Point] Message
```

The following line is an actual error message that is generated if the developer tries to dump the Error Log Management logs to an unallocated buffer:

```
[22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL
```

Where before, all the developer would have gotten is *CUDA\_ERROR\_INVALID\_VALUE* in the return code and possibly “invalid argument” if *cuGetErrorString* is called.