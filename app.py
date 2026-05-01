import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")  # headless backend for Gradio

import matplotlib.pyplot as plt
import gradio as gr
from transformers import AutoTokenizer

from circuit_tracer.graph import Graph, prune_graph
from draw_circuit import draw_circuit


def run(pt_path, node_threshold, edge_threshold, show_error_nodes):
    if pt_path is None:
        return None, "", ""

    graph = Graph.from_pt(pt_path, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)

    node_mask, edge_mask, cumulative_scores = prune_graph(
        graph,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    # Token table
    input_rows = [
        [i, repr(tokenizer.decode(t.item())), t.item()]
        for i, t in enumerate(graph.input_tokens)
    ]
    logit_rows = [
        [repr(tokenizer.decode(tok.item())), round(p.item(), 4)]
        for tok, p in zip(graph.logit_tokens, graph.logit_probabilities)
    ]

    token_md = "### Input tokens\n"
    token_md += " | ".join(f"`{r[1]}`" for r in input_rows) + "\n\n"
    token_md += "### Top logit predictions\n"
    token_md += " ".join(f"`{r[0]}` {r[1]}" for r in logit_rows)

    n_nodes = int(node_mask.sum())
    n_edges = int(edge_mask.sum())
    stats = f"**{n_nodes} nodes · {n_edges} edges** after pruning"

    fig, _ = draw_circuit(
        graph,
        node_mask,
        edge_mask,
        cumulative_scores,
        tokenizer=tokenizer,
        show_error_nodes=show_error_nodes,
    )

    return fig, token_md, stats


with gr.Blocks(title="Circuit Viewer") as demo:
    gr.Markdown("# Circuit Viewer")

    with gr.Row():
        with gr.Column(scale=1):
            pt_file = gr.File(label="Upload .pt graph", file_types=[".pt"])
            node_thresh = gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="Node threshold")
            edge_thresh = gr.Slider(0.0, 1.0, value=0.98, step=0.01, label="Edge threshold")
            error_nodes = gr.Checkbox(label="Show error nodes", value=False)
            run_btn = gr.Button("Draw", variant="primary")

        with gr.Column(scale=3):
            stats_out = gr.Markdown()
            token_out = gr.Markdown()
            plot_out = gr.Plot(label="Circuit")

    run_btn.click(
        fn=run,
        inputs=[pt_file, node_thresh, edge_thresh, error_nodes],
        outputs=[plot_out, token_out, stats_out],
    )

if __name__ == "__main__":
    demo.launch()
