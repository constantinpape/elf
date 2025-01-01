"""Graph-based instance segmentation based on clustering, (lifted) multicut, mutex watershed and more.

Examples for graph-based segmentation with `elf.segmentation` can be found in [example/segmentation](https://github.com/constantinpape/elf/tree/master/example/segmentation) and for embedding-based segmentation in [example/embeddings](https://github.com/constantinpape/elf/tree/master/example/embeddings).
"""

from .clustering import (agglomerative_clustering,
                         cluster_segmentation, cluster_segmentation_mala,
                         mala_clustering)
from .features import (compute_affinity_features, compute_boundary_features, compute_boundary_mean_and_length,
                       compute_grid_graph, compute_grid_graph_affinity_features, compute_grid_graph_image_features,
                       compute_rag, compute_region_features,
                       project_node_labels_to_pixels)
from .lifted_multicut import get_lifted_multicut_solver
from .multicut import get_multicut_solver, compute_edge_costs
from .watershed import distance_transform_watershed, stacked_watershed
from .workflows import edge_training, multicut_segmentation, multicut_workflow, simple_multicut_workflow
from .gasp import GaspFromAffinities, run_GASP
