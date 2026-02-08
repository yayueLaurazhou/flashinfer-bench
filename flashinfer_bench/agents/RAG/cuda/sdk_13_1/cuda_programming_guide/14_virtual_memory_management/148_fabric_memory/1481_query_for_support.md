# 14.8.1. Query for Support

### 14.8.1. Query for Support[](#querying-fabric-mem-support "Permalink to this headline")

Before attempting to use Fabric Memory, applications must ensure that the devices they want to use support Fabric Memory. The following code
sample shows querying for Fabric Memory support:

```
int deviceSupportsFabricMem;
CUresult result = cuDeviceGetAttribute(&deviceSupportsFabricMem, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
if (deviceSupportsFabricMem != 0) {
    // `device` supports Fabric Memory
}
```

Aside from using `CU_MEM_HANDLE_TYPE_FABRIC` as handle type and not requiring OS native mechanisms for inter process communication to exchange
sharable handles there is no difference in using Fabric Memory compared to other allocation handle types.