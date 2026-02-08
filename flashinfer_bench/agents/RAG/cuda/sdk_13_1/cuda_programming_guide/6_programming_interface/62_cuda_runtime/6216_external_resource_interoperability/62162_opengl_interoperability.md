# 6.2.16.2. OpenGL Interoperability

#### 6.2.16.2. OpenGL Interoperability[](#opengl-interoperability-ext-res-int "Permalink to this headline")

Traditional OpenGL-CUDA interop as outlined in [OpenGL Interoperability](#opengl-interoperability) works by CUDA directly consuming handles created in OpenGL. However, since OpenGL can also consume memory and synchronization objects created in Vulkan, there exists an alternative approach to doing OpenGL-CUDA interop. Essentially, memory and synchronization objects exported by Vulkan could be imported into both, OpenGL and CUDA, and then used to coordinate memory accesses between OpenGL and CUDA. Please refer to the following OpenGL extensions for further details on how to import memory and synchronization objects exported by Vulkan:

* GL\_EXT\_memory\_object
* GL\_EXT\_memory\_object\_fd
* GL\_EXT\_memory\_object\_win32
* GL\_EXT\_semaphore
* GL\_EXT\_semaphore\_fd
* GL\_EXT\_semaphore\_win32