# 20.1.2. Family-Specific Features

### 20.1.2. Family-Specific Features[](#family-specific-features "Permalink to this headline")

Beginning with devices of Compute Capability 10.0, some architecture-specific features are common to devices of more than one compute capability. The devices that contain these features are part of the same family and these features can also be called *family-specific* features. Family-specific features are guaranteed to be available on all devices in the same family. A family-specific compiler target is required to enable family-specific features. See [Section 20.1.3](#feature-set-compiler-targets). Code compiled for a family-specific target can only be run on GPUs which are members of that family.