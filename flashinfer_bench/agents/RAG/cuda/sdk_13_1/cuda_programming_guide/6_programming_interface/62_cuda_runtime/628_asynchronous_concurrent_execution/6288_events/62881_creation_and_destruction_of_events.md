# 6.2.8.8.1. Creation and Destruction of Events

##### 6.2.8.8.1. Creation and Destruction of Events[](#creation-and-destruction-of-events "Permalink to this headline")

The following code sample creates two events:

```
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
```

They are destroyed this way:

```
cudaEventDestroy(start);
cudaEventDestroy(stop);
```