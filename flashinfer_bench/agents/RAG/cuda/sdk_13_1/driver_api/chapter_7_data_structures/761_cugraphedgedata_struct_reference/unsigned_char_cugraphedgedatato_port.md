# unsigned char CUgraphEdgeData::to_port

This indicates what portion of the downstream node is dependent on the upstream node or portion
thereof (indicated by from_port). The meaning is specific to the node type. A value of 0 in all cases
means the entirety of the downstream node is dependent on the upstream work. Currently no node
types define non-zero ports. Accordingly, this field must be set to zero.