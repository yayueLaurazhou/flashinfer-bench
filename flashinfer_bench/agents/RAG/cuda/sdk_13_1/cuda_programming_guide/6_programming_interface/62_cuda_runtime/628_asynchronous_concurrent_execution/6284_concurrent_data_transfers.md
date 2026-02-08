# 6.2.8.4. Concurrent Data Transfers

#### 6.2.8.4. Concurrent Data Transfers[](#concurrent-data-transfers "Permalink to this headline")

Some devices of compute capability 2.x and higher can overlap copies to and from the device. Applications may query this capability by checking the `asyncEngineCount` device property (see [Device Enumeration](#device-enumeration)), which is equal to 2 for devices that support it. In order to be overlapped, any host memory involved in the transfers must be page-locked.