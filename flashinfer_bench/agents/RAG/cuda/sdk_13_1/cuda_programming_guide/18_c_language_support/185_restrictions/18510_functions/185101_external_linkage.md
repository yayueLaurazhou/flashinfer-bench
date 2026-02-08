# 18.5.10.1. External Linkage

#### 18.5.10.1. External Linkage[](#external-linkage "Permalink to this headline")

A call within some device code of a function declared with the extern qualifier is only allowed if the function is defined within the same compilation unit as the device code, i.e., a single file or several files linked together with relocatable device code and nvlink.