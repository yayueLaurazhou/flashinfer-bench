# #define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT)

Indicator that the next value in the extra parameter to cuLaunchKernel will be a pointer to a buffer
containing all kernel parameters used for launching kernel f. This buffer needs to honor all alignment/
padding requirements of the individual parameters. If CU_LAUNCH_PARAM_BUFFER_SIZE is not
also specified in the extra array, then CU_LAUNCH_PARAM_BUFFER_POINTER will have no
effect.