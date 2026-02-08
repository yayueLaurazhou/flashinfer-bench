"""
Build a knowledge graph from CUDA documentation using networkx and entity extraction.
Supports GPU acceleration with cugraph for large graphs.
Includes LLM-based querying capabilities for semantic search and reasoning.
"""

import os
import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict
import logging

import networkx as nx
from sentence_transformers import SentenceTransformer

# Try to import cugraph, but make it optional
try:
    import cugraph
    import cudf
    HAS_CUGRAPH = True
except ImportError:
    HAS_CUGRAPH = False    
    import json
    from pathlib import Path
    from typing import List, Dict, Any
    from sentence_transformers import SentenceTransformer
    import logging

# Try to import LangChain for LLM querying
try:
    from langchain.chains import GraphQAChain
    from langchain.indexes.graph import NetworkxEntityGraph
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    ChatNVIDIA = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities from CUDA documentation based on predefined categories."""

    # Entity category patterns
    ENTITY_PATTERNS = {
        "Concept": {
            "keywords": [
                "simt",
                "coalescing",
                "occupancy",
                "bank conflict",
                "divergence",
                "warp",
                "block",
                "grid",
                "memory hierarchy",
                "latency",
                "throughput",
                "bandwidth",
                "pipeline",
                "scheduling",
                "synchronization",
            ],
            "patterns": [
                r"(?:concept|principle|model).*?\b([a-z_]+(?:\s+[a-z_]+)*)\b",
            ],
        },
        "Hardware": {
            "keywords": [
                "streaming multiprocessor",
                "sm",
                "tensor core",
                "l1 cache",
                "l2 cache",
                "global memory",
                "shared memory",
                "register file",
                "warp scheduler",
                "memory controller",
                "interconnect",
                "nvlink",
                "pcie",
            ],
            "patterns": [
                r"(?:hardware|component|device).*?\b([a-z0-9_-]+(?:\s+[a-z0-9_-]+)*)\b",
            ],
        },
        "SoftwareComponent": {
            "keywords": [
                "kernel",
                "thread block",
                "warp",
                "grid",
                "stream",
                "context",
                "event",
                "queue",
                "command buffer",
                "graph",
                "task",
            ],
            "patterns": [
                r"(?:kernel|function|__global__|__device__|__host__)",
            ],
        },
        "Function": {
            "keywords": [
                "cudamalloc",
                "cudafree",
                "cudaMemcpy",
                "syncthreads",
                "blockdim",
                "threadidx",
                "blockidx",
                "gridDim",
                "blockDim",
                "cudaLaunchKernel",
                "cudaDeviceSynchronize",
            ],
            "patterns": [
                r"\b(cuda[A-Z]\w+|__\w+\(\)|\b_\w+)",
                r"(?:__global__|__device__|__host__|__shared__|__constant__)\s+",
            ],
        },
        "Keyword/Type": {
            "keywords": [
                "global",
                "device",
                "host",
                "shared",
                "constant",
                "dim3",
                "float4",
                "int4",
                "uint32_t",
                "uint64_t",
                "nvcc",
                "ptx",
            ],
            "patterns": [
                r"(?:__global__|__device__|__host__|__shared__|__constant__|__managed__|__pinned__)",
            ],
        },
        "Metric": {
            "keywords": [
                "throughput",
                "latency",
                "bandwidth",
                "flops",
                "occupancy",
                "utilization",
                "efficiency",
                "speedup",
                "scalability",
                "memory efficiency",
            ],
            "patterns": [
                r"(?:throughput|latency|bandwidth|flops|occupancy|utilization)",
            ],
        },
        "Tool": {
            "keywords": [
                "nsight systems",
                "nsight compute",
                "cuda-gdb",
                "nvprof",
                "nvtx",
                "cupti",
                "profiler",
                "debugger",
                "visual profiler",
            ],
            "patterns": [
                r"(?:nsight|cuda-gdb|nvprof|nvtx|profiler|debugger)",
            ],
        },
    }

    def __init__(self, model_name: str = "nvidia/NV-Embed-v2"):
        """
        Initialize the entity extractor.

        Args:
            model_name: Model for semantic similarity
        """
        self.model = SentenceTransformer(model_name,trust_remote_code=True)     
        self.extracted_entities: Dict[str, Set[str]] = defaultdict(set)

    def extract_entities_from_text(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract entities from text based on patterns and keywords.

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary mapping entity categories to sets of extracted entities
        """
        entities_found = defaultdict(set)
        text_lower = text.lower()

        for category, patterns_info in self.ENTITY_PATTERNS.items():
            # Match keywords
            for keyword in patterns_info["keywords"]:
                if keyword in text_lower:
                    # Find word boundaries
                    matches = re.finditer(
                        r"\b" + re.escape(keyword) + r"\b", text_lower, re.IGNORECASE
                    )
                    for match in matches:
                        entities_found[category].add(keyword)

            # Match patterns
            for pattern in patterns_info["patterns"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        entity = match.group(1).strip()
                        if len(entity) > 2:
                            entities_found[category].add(entity)

        return entities_found

    def extract_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract entities from a markdown file.

        Args:
            file_path: Path to the markdown file

        Returns:
            Dictionary with file info and extracted entities
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            entities = self.extract_entities_from_text(content)

            # Update global extracted entities
            for category, entity_set in entities.items():
                self.extracted_entities[category].update(entity_set)

            return {
                "file_path": str(file_path),
                "entities": {k: list(v) for k, v in entities.items()},
                "content_length": len(content),
            }
        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {e}")
            return None


class KnowledgeGraphBuilder:
    """Build a knowledge graph from extracted entities and relationships."""

    def __init__(self):
        """Initialize the knowledge graph builder."""
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.file_relationships = []

    def add_entity_node(self, entity: str, category: str, metadata: Dict = None):
        """
        Add an entity node to the graph.

        Args:
            entity: Entity name
            category: Entity category
            metadata: Additional metadata
        """
        node_id = f"{category}:{entity}"
        self.graph.add_node(
            node_id,
            entity=entity,
            category=category,
            metadata=metadata or {},
        )

    def add_relationship(
        self,
        source_entity: str,
        source_category: str,
        target_entity: str,
        target_category: str,
        relationship_type: str,
        metadata: Dict = None,
    ):
        """
        Add a relationship edge between two entities.

        Args:
            source_entity: Source entity
            source_category: Source category
            target_entity: Target entity
            target_category: Target category
            relationship_type: Type of relationship
            metadata: Additional metadata
        """
        source_id = f"{source_category}:{source_entity}"
        target_id = f"{target_category}:{target_entity}"

        # Ensure nodes exist
        if source_id not in self.graph:
            self.add_entity_node(source_entity, source_category)
        if target_id not in self.graph:
            self.add_entity_node(target_entity, target_category)

        # Add edge
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type,
            metadata=metadata or {},
        )

    def extract_relationships(self, entities_by_file: Dict[str, Dict]) -> List[Dict]:
        """
        Extract relationships between entities in the same document.

        Args:
            entities_by_file: Dictionary of entities grouped by file

        Returns:
            List of relationships
        """
        relationships = []

        # Define relationship rules
        relationship_rules = [
            # Hardware <-> Concept relationships
            ("Hardware", "Concept", "uses_principle"),
            # Function <-> Keyword/Type relationships
            ("Function", "Keyword/Type", "implements"),
            # SoftwareComponent <-> Hardware relationships
            ("SoftwareComponent", "Hardware", "executes_on"),
            # Metric <-> SoftwareComponent relationships
            ("Metric", "SoftwareComponent", "measures"),
            # Tool <-> Metric relationships
            ("Tool", "Metric", "measures"),
            # Concept <-> Function relationships
            ("Concept", "Function", "applied_in"),
        ]

        for file_path, entity_dict in entities_by_file.items():
            for source_cat, target_cat, rel_type in relationship_rules:
                source_entities = entity_dict.get(source_cat, [])
                target_entities = entity_dict.get(target_cat, [])

                # Create relationships between all pairs in the same document
                for source_ent in source_entities:
                    for target_ent in target_entities:
                        rel = {
                            "source": source_ent,
                            "source_category": source_cat,
                            "target": target_ent,
                            "target_category": target_cat,
                            "relationship_type": rel_type,
                            "source_file": file_path,
                        }
                        relationships.append(rel)
                        self.add_relationship(
                            source_ent, source_cat, target_ent, target_cat, rel_type,
                            {"source_file": file_path}
                        )

        return relationships

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        nodes_by_category = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            category = data.get("category", "unknown")
            nodes_by_category[category] += 1

        edges_by_type = defaultdict(int)
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get("relationship_type", "unknown")
            edges_by_type[rel_type] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_category": dict(nodes_by_category),
            "edges_by_type": dict(edges_by_type),
            "density": nx.density(self.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
        }

    def find_entity_connections(self, entity: str, category: str, depth: int = 2) -> Dict:
        """
        Find connected entities (semantic neighborhood).

        Args:
            entity: Entity name
            category: Entity category
            depth: Depth of connections to find

        Returns:
            Dictionary of connected entities
        """
        node_id = f"{category}:{entity}"
        if node_id not in self.graph:
            return {}

        connections = {
            "predecessors": [],
            "successors": [],
            "all_neighbors": [],
        }

        # Get successors
        for successor in nx.descendants(self.graph, node_id):
            connections["successors"].append(successor)

        # Get predecessors
        for predecessor in nx.ancestors(self.graph, node_id):
            connections["predecessors"].append(predecessor)

        # Get all neighbors
        for neighbor in nx.all_neighbors(self.graph, node_id):
            connections["all_neighbors"].append(neighbor)

        return connections

    def save_graph(self, output_path: Path):
        """
        Save the knowledge graph to a file.

        Args:
            output_path: Path to save the graph
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as GraphML (preserves all attributes)
        # GraphML doesn't support dict values, so serialize metadata to JSON strings
        graphml_graph = self.graph.copy()
        for node, data in graphml_graph.nodes(data=True):
            if "metadata" in data and isinstance(data["metadata"], dict):
                graphml_graph.nodes[node]["metadata"] = json.dumps(data["metadata"])
        for u, v, data in graphml_graph.edges(data=True):
            if "metadata" in data and isinstance(data["metadata"], dict):
                graphml_graph.edges[u, v]["metadata"] = json.dumps(data["metadata"])

        graphml_path = output_path.with_suffix(".graphml")
        nx.write_graphml(graphml_graph, graphml_path)
        logger.info(f"Graph saved to {graphml_path}")

        # Save as pickle for Python usage
        pickle_path = output_path.with_suffix(".pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info(f"Graph pickled to {pickle_path}")

        # Save statistics
        stats_path = output_path.with_suffix(".json")
        stats = self.get_graph_statistics()
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_path}")

    def load_graph(self, graph_path: Path):
        """
        Load a saved knowledge graph.

        Args:
            graph_path: Path to the saved graph
        """
        pickle_path = Path(graph_path).with_suffix(".pkl")
        if pickle_path.exists():
            with open(pickle_path, "rb") as f:
                self.graph = pickle.load(f)
            logger.info(f"Graph loaded from {pickle_path}")
        else:
            logger.error(f"Graph file not found: {pickle_path}")

    def to_cugraph(self):
        """
        Convert NetworkX graph to cuGraph for GPU processing.

        Returns:
            cuGraph Graph object or None if cugraph not available
        """
        if not HAS_CUGRAPH:
            logger.warning("cuGraph not installed. Returning NetworkX graph.")
            return self.graph

        try:
            # Create edge list
            edges = []
            for source, target, data in self.graph.edges(data=True):
                edges.append((source, target))

            if not edges:
                logger.warning("No edges in graph")
                return None

            # Convert to cuDF
            edge_df = cudf.DataFrame(edges, columns=["source", "target"])

            # Create cuGraph
            cu_graph = cugraph.Graph()
            cu_graph.from_cudf_edgelist(
                edge_df, source="source", target="target", renumber=True
            )

            logger.info(f"Converted to cuGraph: {cu_graph.number_of_nodes()} nodes, {cu_graph.number_of_edges()} edges")
            return cu_graph
        except Exception as e:
            logger.error(f"Error converting to cuGraph: {e}")
            return None


class GraphRAGBuilder:
    """Main class to coordinate graph RAG construction."""

    def __init__(self, cuda_folder: str, output_dir: str = "./knowledge_base"):
        """
        Initialize the graph RAG builder.

        Args:
            cuda_folder: Path to CUDA documentation folder
            output_dir: Directory to save outputs
        """
        self.cuda_folder = Path(cuda_folder)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = EntityExtractor()
        self.builder = KnowledgeGraphBuilder()
        self.entities_by_file = {}

    def build(self):
        """Build the complete knowledge graph."""
        logger.info("Starting graph RAG construction...")

        # Step 1: Extract entities from all markdown files
        logger.info("Step 1: Extracting entities from documents...")
        self._extract_entities()

        # Step 2: Build the knowledge graph
        logger.info("Step 2: Building knowledge graph...")
        self._build_graph()

        # Step 3: Save the graph
        logger.info("Step 3: Saving graph...")
        self._save_outputs()

        logger.info("Graph RAG construction complete!")

    def _extract_entities(self):
        """Extract entities from all markdown files."""
        md_files = list(self.cuda_folder.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")

        for file_path in md_files:
            result = self.extractor.extract_from_file(file_path)
            if result:
                self.entities_by_file[str(file_path)] = result["entities"]

        logger.info(
            f"Extracted entities: {dict(self.extractor.extracted_entities)}"
        )

    def _build_graph(self):
        """Build the knowledge graph from extracted entities."""
        # Add all entities as nodes
        for category, entities in self.extractor.extracted_entities.items():
            for entity in entities:
                self.builder.add_entity_node(entity, category)

        logger.info(f"Added {self.builder.graph.number_of_nodes()} entity nodes")

        # Extract and add relationships
        relationships = self.builder.extract_relationships(self.entities_by_file)
        logger.info(f"Added {len(relationships)} relationships")

        # Print statistics
        stats = self.builder.get_graph_statistics()
        logger.info(f"Graph statistics: {json.dumps(stats, indent=2)}")

    def _save_outputs(self):
        """Save the graph and related outputs."""
        # Save graph
        graph_output = self.output_dir / "cuda_knowledge_graph"
        self.builder.save_graph(graph_output)

        # Save extracted entities
        entities_output = self.output_dir / "entities.json"
        entities_dict = {
            k: list(v) for k, v in self.extractor.extracted_entities.items()
        }
        with open(entities_output, "w") as f:
            json.dump(entities_dict, f, indent=2)
        logger.info(f"Entities saved to {entities_output}")

        # Try to convert to cuGraph
        if HAS_CUGRAPH:
            logger.info("Converting to cuGraph...")
            cu_graph = self.builder.to_cugraph()
            if cu_graph:
                logger.info("Successfully converted to cuGraph")

    def query_entity(self, entity: str, category: str):
        """
        Query the knowledge graph for an entity.

        Args:
            entity: Entity name
            category: Entity category
        """
        connections = self.builder.find_entity_connections(entity, category)
        logger.info(f"Connections for {category}:{entity}:")
        logger.info(f"  Predecessors: {connections.get('predecessors', [])}")
        logger.info(f"  Successors: {connections.get('successors', [])}")

    def query_with_llm(self, query: str, llm=None):
        """
        Query the knowledge graph using an LLM.

        Args:
            query: Natural language query
            llm: LLM instance (will use ChatNVIDIA if not provided)

        Returns:
            Query result from the LLM
        """
        if not HAS_LANGCHAIN:
            logger.error("LangChain not installed. Cannot query with LLM.")
            return None

        # Initialize LLM if not provided
        if llm is None:
            if ChatNVIDIA is None:
                logger.error("ChatNVIDIA not available. Please install langchain-nvidia-ai-endpoints.")
                return None
            llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct")

        try:
            # Create NetworkX entity graph for LangChain
            entity_graph = NetworkxEntityGraph(self.builder.graph)
            
            # Create the QA chain
            chain = GraphQAChain.from_llm(llm=llm, graph=entity_graph, verbose=True)
            
            # Run the query
            logger.info(f"Querying: {query}")
            result = chain.run(query)
            
            return result
        except Exception as e:
            logger.error(f"Error querying with LLM: {e}")
            return None


def main():
    """Main function to build the graph RAG."""
    # Get the cuda folder path
    current_dir = Path(__file__).parent
    cuda_folder = current_dir / "cuda"

    if not cuda_folder.exists():
        logger.error(f"Cuda folder not found at {cuda_folder}")
        return

    # Build the graph RAG
    builder = GraphRAGBuilder(
        cuda_folder=str(cuda_folder),
        output_dir=str(current_dir / "knowledge_base"),
    )

    builder.build()

    # Example entity queries
    logger.info("\n=== Example Entity Queries ===")
    builder.query_entity("occupancy", "Concept")
    builder.query_entity("global memory", "Hardware")
    builder.query_entity("kernel", "SoftwareComponent")

    # Example LLM queries
    logger.info("\n=== Example LLM Queries ===")
    if HAS_LANGCHAIN:
        try:
            # Initialize the LLM once
            llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct")
            
            # Example queries
            queries = [
                "Explain how kernel execution relates to hardware components.",
                "What tools can measure bandwidth and how do they relate to memory?",
            ]
            
            for query in queries:
                logger.info(f"\nQuery: {query}")
                result = builder.query_with_llm(query, llm)
                if result:
                    logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error with LLM queries: {e}")
    else:
        logger.warning("LangChain not installed. Skipping LLM queries. Install with: pip install langchain langchain-nvidia-ai-endpoints")


if __name__ == "__main__":
    main()
