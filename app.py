import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

import gradio as gr
from transformers import AutoTokenizer

from circuit_tracer.graph import Graph, prune_graph
from draw_circuit import draw_circuit


def find_pt_files(graphs_dir: str) -> list[str]:
    d = Path(graphs_dir)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.rglob("*.pt"))


def run(pt_path, node_threshold, edge_threshold, show_error_nodes):
    if not pt_path:
        return None, "", ""

    graph = Graph.from_pt(pt_path, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)

    node_mask, edge_mask, cumulative_scores = prune_graph(
        graph,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    token_md = "### Input tokens\n"
    token_md += " | ".join(
        f"`{tokenizer.decode(t.item())}`" for t in graph.input_tokens
    ) + "\n\n"
    token_md += "### Top logit predictions\n"
    token_md += " ".join(
        f"`{tokenizer.decode(tok.item())}` {p.item():.3f}"
        for tok, p in zip(graph.logit_tokens, graph.logit_probabilities)
    )

    n_nodes = int(node_mask.sum())
    n_edges = int(edge_mask.sum())
    stats = f"**{n_nodes} nodes · {n_edges} edges** after pruning"

    fig, _ = draw_circuit(
        graph, node_mask, edge_mask, cumulative_scores,
        tokenizer=tokenizer,
        show_error_nodes=show_error_nodes,
    )

    return fig, token_md, stats


def build_ui(graphs_dir: str):
    pt_files = find_pt_files(graphs_dir)

    with gr.Blocks(title="Circuit Viewer") as demo:
        gr.Markdown("# Circuit Viewer")

        with gr.Row():
            with gr.Column(scale=1):
                if pt_files:
                    pt_input = gr.Dropdown(
                        choices=pt_files,
                        label="Graph file",
                        info=f"Scanning: {graphs_dir}",
                    )
                    refresh_btn = gr.Button("Refresh file list")

                    def refresh():
                        return gr.Dropdown(choices=find_pt_files(graphs_dir))

                    refresh_btn.click(fn=refresh, outputs=pt_input)
                else:
                    pt_input = gr.Textbox(
                        label="Path to .pt file",
                        placeholder="/path/to/graph.pt",
                    )

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
            inputs=[pt_input, node_thresh, edge_thresh, error_nodes],
            outputs=[plot_out, token_out, stats_out],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphs-dir", default="graphs",
        help="Directory to scan for .pt files (default: ./graphs)",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(args.graphs_dir)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
