import torch
from circuit_tracer import ReplacementModel, attribute
from pathlib import Path

model_name = "Qwen/Qwen3-4B"
transcoder_name = "mwhanna/qwen3-4b-transcoders"
backend = "nnsight"

model = ReplacementModel.from_pretrained(
    model_name,
    transcoder_name,
    dtype=torch.bfloat16,
    backend=backend,
)

prompt = "The capital of state containing Dallas is"  # What you want to get the graph for
max_n_logits = 10  # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter
desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
max_feature_nodes = 8192  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
batch_size = 256  # Batch size when attributing
offload = "cpu"
# (    "disk" if IN_COLAB else "cpu")  # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)
verbose = True  # Whether to display a tqdm progress bar and timing report

graph = attribute(
    prompt=prompt,
    model=model,
    max_n_logits=max_n_logits,
    desired_logit_prob=desired_logit_prob,
    batch_size=batch_size,
    max_feature_nodes=max_feature_nodes,
    offload=offload,
    verbose=verbose,
)

graph_dir = "home/rds/graphs"
graph_name = "dallas.pt"
graph_dir = Path(graph_dir)
graph_dir.mkdir(exist_ok=True)
graph_path = graph_dir / graph_name

graph.to_pt(graph_path)