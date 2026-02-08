# 13.6.3.3.1.1. Memory Footprint (CDP1)

###### 13.6.3.3.1.1. Memory Footprint (CDP1)[](#memory-footprint-cdp1 "Permalink to this headline")

See [Memory Footprint](#memory-footprint), above, for CDP2 version of document.

The device runtime system software reserves memory for various management purposes, in particular one reservation which is used for saving parent-grid state during synchronization, and a second reservation for tracking pending grid launches. Configuration controls are available to reduce the size of these reservations in exchange for certain launch limitations. See [Configuration Options (CDP1)](#configuration-options-cdp1), below, for details.

The majority of reserved memory is allocated as backing-store for parent kernel state, for use when synchronizing on a child launch. Conservatively, this memory must support storing of state for the maximum number of live threads possible on the device. This means that each parent generation at which `cudaDeviceSynchronize()` is callable may require up to 860MB of device memory, depending on the device configuration, which will be unavailable for program use even if it is not all consumed.