# Complexity vs. control

The runtime API eases device code management by providing implicit primary context initialization
and management, and implicit module management. This leads to simpler code, but it also lacks the
level of control that the driver API has.

In comparison, the driver API offers more fine-grained control, especially over module loading. Kernel
launches are much more complex to implement, as the execution configuration and kernel parameters
must be specified with explicit function calls. However, unlike the runtime, where all the kernels are
automatically loaded during initialization and stay loaded for as long as the program runs, with the
driver API it is possible to only keep the modules that are currently needed loaded, or even dynamically
reload modules. The driver API is also language-independent as it only deals with cubin objects.