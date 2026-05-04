import torch
from circuit_tracer import ReplacementModel, attribute
from pathlib import Path
from circuit_tracer.graph import Graph, prune_graph
from draw_circuit import draw_circuit, draw_circuit_from_pt
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# import
graph = Graph.from_pt(Path(os.environ["HOME"]) / "rds/graphs/example_graph.pt")
print("Imported original graph")

# pruning
# pruned_graph = prune_graph(graph,  node_threshold=0.8, edge_threshold=0.98) 
# print("Pruned graph")

node_mask, edge_mask, cumulative_scores = prune_graph(graph, node_threshold=0.75, edge_threshold=0.75)
fig, ax = draw_circuit(graph, node_mask, edge_mask, cumulative_scores)
plt.savefig(Path(os.environ["HOME"]) / "rds/plots/example_graph2(75).png")
