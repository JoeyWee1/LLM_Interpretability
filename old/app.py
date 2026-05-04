import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from transformers import AutoTokenizer

from circuit_tracer.graph import Graph, prune_graph
from Thesis.LLM_Interpretability.old.plotly_circuit import build_plotly_circuit
from Thesis.LLM_Interpretability.old.feature_fetch import fetch_feature, format_feature_html


def find_pt_files(graphs_dir: str) -> list[str]:
    d = Path(graphs_dir)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.rglob("*.pt"))


def load_graph(pt_path, node_threshold, edge_threshold, show_error_nodes):
    if not pt_path:
        return None, None, "", ""

    graph = Graph.from_pt(pt_path, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)

    node_mask, edge_mask, cumulative_scores = prune_graph(
        graph, node_threshold=node_threshold, edge_threshold=edge_threshold,
    )

    fig, node_metadata = build_plotly_circuit(
        graph, node_mask, edge_mask, cumulative_scores,
        tokenizer=tokenizer,
        show_error_nodes=show_error_nodes,
    )

    token_md = "**Input:** " + " ".join(
        f"`{tokenizer.decode(t.item())}`" for t in graph.input_tokens
    ) + "\n\n"
    token_md += "**Logits:** " + " ".join(
        f"`{tokenizer.decode(tok.item())}` {p.item():.3f}"
        for tok, p in zip(graph.logit_tokens, graph.logit_probabilities)
    )

    n_nodes = int(node_mask.sum())
    n_edges = int(edge_mask.sum())
    stats = f"**{n_nodes} nodes · {n_edges} edges**"

    return fig, node_metadata, token_md, stats


def on_node_click(evt: gr.SelectData, node_metadata):
    if evt is None or node_metadata is None:
        return "<p>Click a node to see details.</p>"

    idx = evt.index
    if not isinstance(idx, int) or idx >= len(node_metadata):
        return "<p>Click a node to see details.</p>"

    meta = node_metadata[idx]
    ntype = meta["type"]

    if ntype == "token":
        return f"<p><b>Input token</b> at position {meta['pos']}</p>"

    if ntype == "logit":
        return (
            f"<p><b>Logit:</b> <code>{meta['logit_token']}</code><br>"
            f"<b>P =</b> {meta['logit_prob']:.4f}</p>"
        )

    if ntype == "error":
        return f"<p><b>Residual error node</b> — Layer {meta['layer']}, Position {meta['pos']}</p>"

    # Feature node — fetch from HuggingFace
    layer, feat_idx, scan = meta["layer"], meta["feat_idx"], meta["scan"]
    header = (
        f"<p><b>Feature {feat_idx}</b> · Layer {layer} · Position {meta['pos']}</p>"
    )
    if not scan:
        return header + "<p><i>No scan ID — cannot fetch examples.</i></p>"

    data = fetch_feature(scan, layer, feat_idx)
    return header + format_feature_html(data)


def build_ui(graphs_dir: str):
    pt_files = find_pt_files(graphs_dir)

    with gr.Blocks(title="Circuit Viewer") as demo:
        gr.Markdown("# Circuit Viewer")
        node_metadata_state = gr.State(None)

        with gr.Row():
            # --- Left panel: controls ---
            with gr.Column(scale=1, min_width=220):
                if pt_files:
                    pt_input = gr.Dropdown(choices=pt_files, label="Graph file")
                    refresh_btn = gr.Button("↻ Refresh", size="sm")
                    refresh_btn.click(
                        fn=lambda: gr.Dropdown(choices=find_pt_files(graphs_dir)),
                        outputs=pt_input,
                    )
                else:
                    pt_input = gr.Textbox(label="Path to .pt file", placeholder="/path/to/graph.pt")

                node_thresh = gr.Slider(0.0, 1.0, value=0.8,  step=0.01, label="Node threshold")
                edge_thresh = gr.Slider(0.0, 1.0, value=0.98, step=0.01, label="Edge threshold")
                error_nodes = gr.Checkbox(label="Show error nodes", value=False)
                draw_btn = gr.Button("Draw", variant="primary")

                stats_out = gr.Markdown()
                token_out = gr.Markdown()

            # --- Right panel: graph + feature detail ---
            with gr.Column(scale=3):
                plot_out = gr.Plot(label="Circuit")
                feature_panel = gr.HTML(
                    value="<p style='color:#888'>Click a feature node to see activating examples.</p>",
                    label="Feature detail",
                )

        draw_btn.click(
            fn=load_graph,
            inputs=[pt_input, node_thresh, edge_thresh, error_nodes],
            outputs=[plot_out, node_metadata_state, token_out, stats_out],
        )

        plot_out.select(
            fn=on_node_click,
            inputs=[node_metadata_state],
            outputs=[feature_panel],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-dir", default="graphs")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(args.graphs_dir)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
