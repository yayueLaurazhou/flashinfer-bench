# 23.2. Activation

## 23.2. Activation[](#activation "Permalink to this headline")

Set the *CUDA\_LOG\_FILE* environment variable. Acceptable values are *stdout*, *stderr*, or a valid path on the system to write a file.
The log buffer can be dumped via API even if *CUDA\_LOG\_FILE* was not set before program execution.
NOTE: An error-free execution may not print any logs.