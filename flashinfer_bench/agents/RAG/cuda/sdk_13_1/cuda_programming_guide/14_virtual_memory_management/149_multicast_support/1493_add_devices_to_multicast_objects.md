# 14.9.3. Add Devices to Multicast Objects

### 14.9.3. Add Devices to Multicast Objects[](#add-devices-to-multicast-objects "Permalink to this headline")

Devices can be added to a Multicast Team with `cuMulticastAddDevice`:

```
cuMulticastAddDevice(&mcHandle, device);
```

This step needs to be completed on all processes controlling devices that should participate in a Multicast Team before memory on any device is
bound to the Multicast Object.