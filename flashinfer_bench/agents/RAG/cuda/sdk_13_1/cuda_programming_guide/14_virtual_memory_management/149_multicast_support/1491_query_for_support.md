# 14.9.1. Query for Support

### 14.9.1. Query for Support[](#querying-multicast-obj-mem-support "Permalink to this headline")

Before attempting to use Multicast Objects, applications must ensure that the devices they want to use support them. The following code sample
shows querying for Fabric Memory support:

```
int deviceSupportsMultiCast;
CUresult result = cuDeviceGetAttribute(&deviceSupportsMultiCast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
if (deviceSupportsMultiCast != 0) {
    // `device` supports Multicast Objects
}
```