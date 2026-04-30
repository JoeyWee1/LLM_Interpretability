import torch
from circuit_tracer import ReplacementModel, attribute
from pathlib import Path
from circuit_tracer.graph import Graph
import os


graph = Graph.from_pt(Path(os.environ["HOME"]) / "rds/graphs/example_graph.pt")
