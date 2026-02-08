# 6.2.1. Initialization

### 6.2.1. Initialization[](#initialization "Permalink to this headline")

As of CUDA 12.0, the `cudaInitDevice()` and `cudaSetDevice()` calls initialize the runtime and the primary context
associated with the specified device. Absent these calls, the runtime will implicitly use device 0 and self-initialize
as needed to process other runtime API requests. One needs to keep this in mind when timing runtime function calls and
when interpreting the error code from the first call into the runtime. Before 12.0, `cudaSetDevice()` would not
initialize the runtime and applications would often use the no-op runtime call `cudaFree(0)` to isolate the runtime
initialization from other api activity (both for the sake of timing and error handling).

The runtime creates a CUDA context for each device in the system (see [Context](#context) for more details on CUDA contexts). This
context is the *primary context* for this device and is initialized at the first runtime function which requires an active
context on this device. It is shared among all the host threads of the application. As part of this context creation, the
device code is just-in-time compiled if necessary (see [Just-in-Time Compilation](#just-in-time-compilation)) and loaded into device memory. This
all happens transparently. If needed, for example, for driver API interoperability, the primary context of a device can be
accessed from the driver API as described in [Interoperability between Runtime and Driver APIs](#interoperability-between-runtime-and-driver-apis).

When a host thread calls `cudaDeviceReset()`, this destroys the primary context of the device the host thread currently
operates on (that is, the current device as defined in [Device Selection](#device-selection)). The next runtime function call made by any
host thread that has this device as current will create a new primary context for this device.

Note

The CUDA interfaces use global state that is initialized during host program initiation and destroyed during host program termination. The CUDA runtime and driver cannot detect if this state is invalid, so using any of these interfaces (implicitly or explicitly) during program initiation or termination after main) will result in undefined behavior.

As of CUDA 12.0, `cudaSetDevice()` will now explicitly initialize the runtime after changing the current device for the host thread. Previous versions of CUDA delayed runtime initialization on the new device until the first runtime call was made after `cudaSetDevice()`. This change means that it is now very important to check the return value of `cudaSetDevice()` for initialization errors.

The runtime functions from the error handling and version management sections of the reference manual do not initialize the runtime.