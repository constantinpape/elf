from elf.segmentation.multicut import multicut_rama
from elf.segmentation.utils import load_multicut_problem

graph, costs = load_multicut_problem("A", "small")
node_labels = multicut_rama(graph, costs)
print("Number of nodes in result:", len(node_labels))
