# 13.2.2.1.3. Constant Memory

##### 13.2.2.1.3. Constant Memory[](#constant-memory "Permalink to this headline")

Constants may not be modified from the device. They may only be modified from the host, but the behavior of modifying a constant from the host while there is a concurrent grid that access that constant at any point during its lifetime is undefined.