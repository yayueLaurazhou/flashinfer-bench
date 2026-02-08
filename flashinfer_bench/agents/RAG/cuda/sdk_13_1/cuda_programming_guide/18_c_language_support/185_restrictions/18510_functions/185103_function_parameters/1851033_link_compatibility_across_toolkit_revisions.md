# 18.5.10.3.3. Link Compatibility across Toolkit Revisions

##### 18.5.10.3.3. Link Compatibility across Toolkit Revisions[](#link-compatibility-across-toolkit-revisions "Permalink to this headline")

When linking device objects, if at least one device object contains a kernel with a parameter larger than 4KB, the developer must recompile all objects from their respective device sources with the 12.1 toolkit or higher before linking them together. Failure to do so will result in a linker error.