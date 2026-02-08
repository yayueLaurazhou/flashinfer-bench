# #define CU_LAUNCH_PARAM_BUFFER_SIZE ((void*)CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT)

Indicator that the next value in the extra parameter to cuLaunchKernel will be a pointer to a size_t
which contains the size of the buffer specified with CU_LAUNCH_PARAM_BUFFER_POINTER. It
is required that CU_LAUNCH_PARAM_BUFFER_POINTER also be specified in the extra array if
the value associated with CU_LAUNCH_PARAM_BUFFER_SIZE is not zero.