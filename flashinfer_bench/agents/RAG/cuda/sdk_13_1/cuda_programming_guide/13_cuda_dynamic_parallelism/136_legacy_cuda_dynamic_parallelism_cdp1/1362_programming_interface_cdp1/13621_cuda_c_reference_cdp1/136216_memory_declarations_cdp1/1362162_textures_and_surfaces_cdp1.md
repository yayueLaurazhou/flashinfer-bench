# 13.6.2.1.6.2. Textures and Surfaces (CDP1)

###### 13.6.2.1.6.2. Textures and Surfaces (CDP1)[](#textures-and-surfaces-cdp1 "Permalink to this headline")

See [Textures and Surfaces](#textures-and-surfaces), above, for CDP2 version of document.

CUDA supports dynamically created texture and surface objects[7](#fn14), where a texture object may be created on the host, passed to a kernel, used by that kernel, and then destroyed from the host. The device runtime does not allow creation or destruction of texture or surface objects from within device code, but texture and surface objects created from the host may be used and passed around freely on the device. Regardless of where they are created, dynamically created texture objects are always valid and may be passed to child kernels from a parent.

Note

The device runtime does not support legacy module-scope (i.e., Fermi-style) textures and surfaces within a kernel launched from the device. Module-scope (legacy) textures may be created from the host and used in device code as for any kernel, but may only be used by a top-level kernel (i.e., the one which is launched from the host).