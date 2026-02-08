# 7.2. cudaAccessPolicyWindow Struct Reference

Specifies an access policy for a window, a contiguous extent of memory beginning at base_ptr and
ending at base_ptr + num_bytes. Partition into many segments and assign segments such that. sum
of "hit segments" / window == approx. ratio. sum of "miss segments" / window == approx 1-ratio.
Segments and ratio specifications are fitted to the capabilities of the architecture. Accesses in a hit
segment apply the hitProp access policy. Accesses in a miss segment apply the missProp access policy.