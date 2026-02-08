# 20.1.3. Feature Set Compiler Targets

### 20.1.3. Feature Set Compiler Targets[](#feature-set-compiler-targets "Permalink to this headline")

There are three sets of compute features which the compiler can target:

**Baseline Feature Set**: The predominant set of compute features that are introduced with the intent to be available for subsequent compute architectures. These features and their availability are summarized in [Table 26](#features-and-technical-specifications-feature-support-per-compute-capability).

**Architecture-Specific Feature Set**: A small and highly specialized set of features called architecture-specific, that are introduced to accelerate specialized operations, which are not guaranteed to be available or might change significantly on subsequent compute architectures. These features are summarized in the respective “Compute Capability #.#” subsections. The architecture-specific feature set is a superset of the family-specific feature set. Architecture-specific compiler targets were introduced with Compute Capability 9.0 devices and are selected by using an **a** suffix in the compilation target, for example by specifying `compute_100a` or `compute_120a` as the compute target.

**Family-Specific Feature Set**: Some architecture-specific features are common to GPUs of more than one compute capability. These features are summarized in the respective “Compute Capability #.#” subsections. With a few exceptions, later generation devices with the same major compute capability are in the same family. [Table 25](#family-specific-compatibility) indicates the compatibility of family-specific targets with device compute capability, including exceptions. The family-specific feature set is a superset of the baseline feature set. Family-specific compiler targets were introduced with Compute Capability 10.0 devices and are selected by using a **f** suffix in the compilation target, for example by specifying `compute_100f` or `compute_120f` as the compute target.

All devices starting from compute capability 9.0 have a set of features that are architecture-specific. To utilize the complete set of these features on a specific GPU, the architecture-specific compiler target with the suffix **a** must be used. Additionally, starting from compute capability 10.0, there are sets of features that appear in multiple devices with different minor compute capability. These sets of instructions are called family-specific features, and the devices which share these features are said to be part of the same family. The family-specific features are a subset of the architecture-specific features that are shared by all members of that GPU family. The family-specific compiler target with the suffix **f** allows the compiler to generate code which uses this common subset of architecture-specific features.

For example:

* The `compute_100` compilation target does not allow use of architecture-specific features. This target will be compatible with all devices of compute capability 10.0 and later.
* The `compute_100f` *family-specific* compilation target allows the use of the subset of architecture-specific features that are common across the GPU family. This target will only be compatible with devices that are part of the GPU family. In this example it is compatible with devices of Compute Capability 10.0 and Compute Capability 10.3. The features available in the family-specific `compute_100f` target is a superset of the features available in the baseline `compute_100` target.
* The `compute_100a` *architecture-specific* compilation target allows use of the complete set of architecture-specific features in Compute Capability 10.0 devices. This target will only be compatible with devices of Compute Capability 10.0 and no others. The features available in the `compute_100a` target form a superset of the features available in the `compute_100f` target.

Table 25 Family-Specific Compatibility[](#family-specific-compatibility "Permalink to this table")





| Compilation Target | Compatible with Compute Capability | |
| --- | --- | --- |
| `compute_100f` | 10.0 | 10.3 |
| `compute_103f` | 10.3 [26](#family2) | |
| `compute_110f` | 11.0 [26](#family2) | |
| `compute_120f` | 12.0 | 12.1 |
| `compute_121f` | 12.1 [26](#family2) | |

26([1](#id395),[2](#id396),[3](#id397))
:   Some families only contain a single member when they are created. They may be expanded in the future to include more devices.