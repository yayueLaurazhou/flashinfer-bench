# 6.2.8.7.8.4. Conditional WHILE Nodes

###### 6.2.8.7.8.4. Conditional WHILE Nodes[](#conditional-while-nodes "Permalink to this headline")

The body graph of a WHILE node will be executed as long as the condition is non-zero. The condition will be
evaluated when the node is executed and after completion of the body graph. The following diagram depicts
a 3 node graph where the middle node, B, is a conditional node:

![_images/conditional-while-node.png](_images/conditional-while-node.png)


Figure 24 Conditional WHILE Node[](#id466 "Permalink to this image")

The following code illustrates the creation of a graph containing a WHILE conditional node. The handle
is created using *cudaGraphCondAssignDefault* to avoid the need for an upstream kernel. The body of the
conditional is populated using the [graph API](#creating-a-graph-using-graph-apis).

```
__global__ void loopKernel(cudaGraphConditionalHandle handle)
{
    static int count = 10;
    cudaGraphSetConditional(handle, --count ? 1 : 0);
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[1];

    cuGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeWhile;
    cParams.conditional.size   = 1;
    cudaGraphAddNode(&node, graph, NULL, NULL, 0, &cParams);

    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)loopKernel;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    cudaGraphAddNode(&node, bodyGraph, NULL, NULL, 0, &params);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```