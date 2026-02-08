# enum CUmem_advise

Memory advise values

###### Values

**CU_MEM_ADVISE_SET_READ_MOSTLY = 1**
Data will mostly be read and only occasionally be written to
**CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2**
Undo the effect of CU_MEM_ADVISE_SET_READ_MOSTLY
**CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3**
Set the preferred location for the data as the specified device
**CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4**
Clear the preferred location for the data
**CU_MEM_ADVISE_SET_ACCESSED_BY = 5**
Data will be accessed by the specified device, so prevent page faults as much as possible
**CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6**
Let the Unified Memory subsystem decide on the page faulting policy for the specified device