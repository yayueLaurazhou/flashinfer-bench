# 6.2.8.7.2. Creating a Graph Using Graph APIs

##### 6.2.8.7.2. Creating a Graph Using Graph APIs[](#creating-a-graph-using-graph-apis "Permalink to this headline")

Graphs can be created via two mechanisms: explicit API and stream capture. The following is an example of creating and executing the below graph.

[![Creating a Graph Using Graph APIs Example](_images/create-a-graph.png)](_images/create-a-graph.png)


Figure 14 Creating a Graph Using Graph APIs Example[](#creating-a-graph-using-api-fig-creating-using-graph-apis "Permalink to this image")

```
// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);

// For the purpose of this example, we'll create
// the nodes separately from the dependencies to
// demonstrate that it can be done in two stages.
// Note that dependencies can also be specified
// at node creation.
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

// Now set up dependencies on each node
cudaGraphAddDependencies(graph, &a, &b, NULL, 1);     // A->B
cudaGraphAddDependencies(graph, &a, &c, NULL, 1);     // A->C
cudaGraphAddDependencies(graph, &b, &d, NULL, 1);     // B->D
cudaGraphAddDependencies(graph, &c, &d, NULL, 1);     // C->D
```