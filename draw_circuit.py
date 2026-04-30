import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from circuit_tracer.graph import Graph, prune_graph


NODE_COLORS = {
    "feature": "#4C72B0",
    "error":   "#DD8452",
    "token":   "#55A868",
    "logit":   "#C44E52",
}

NODE_RADIUS = 0.18


def _decode_node(graph: Graph, idx: int):
    """Return (type, layer, pos, extra) for a node index.

    type  : 'feature' | 'error' | 'token' | 'logit'
    layer : actual model layer (-1 for tokens, n_layers for logits)
    pos   : token position
    extra : feat_idx (feature) or logit_rank (logit), else None
    """
    n_feat = len(graph.selected_features)
    n_layers = graph.cfg.n_layers
    n_pos = graph.n_pos

    error_end = n_feat + n_layers * n_pos
    token_end = error_end + n_pos

    if idx < n_feat:
        layer, pos, feat_idx = graph.active_features[graph.selected_features[idx]].tolist()
        return "feature", int(layer), int(pos), int(feat_idx)
    elif idx < error_end:
        layer, pos = divmod(idx - n_feat, n_pos)
        return "error", layer, pos, None
    elif idx < token_end:
        return "token", -1, idx - error_end, None
    else:
        logit_rank = idx - token_end
        return "logit", n_layers, n_pos - 1, logit_rank


def draw_circuit(
    graph: Graph,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    cumulative_scores: torch.Tensor,
    tokenizer=None,
    show_error_nodes: bool = False,
    figsize=None,
    title: str | None = None,
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw the pruned circuit graph.

    Layout
    ------
    X-axis  : token position
    Y-axis  : model layer (compact — only layers with active nodes shown)

    Node colours
    ------------
    Blue    : transcoder feature
    Green   : input token / embedding
    Red     : output logit
    Orange  : residual-stream error (hidden by default)

    Edge colours
    ------------
    Blue    : positive direct effect
    Red     : negative direct effect
    Opacity / width scale with |weight|.

    Parameters
    ----------
    graph              : Graph object (loaded from .pt)
    node_mask          : bool tensor from prune_graph
    edge_mask          : bool tensor from prune_graph
    cumulative_scores  : float tensor from prune_graph
    tokenizer          : optional HuggingFace tokenizer for token labels
    show_error_nodes   : whether to include error nodes
    figsize            : matplotlib figsize (auto-computed if None)
    title              : plot title (defaults to the prompt string)
    save_path          : if given, save the figure here

    Returns
    -------
    (fig, ax)
    """
    n_feat = len(graph.selected_features)
    n_layers = graph.cfg.n_layers
    n_pos = graph.n_pos
    error_end = n_feat + n_layers * n_pos

    # Collect kept node indices
    kept = node_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(kept, int):
        kept = [kept]
    if not show_error_nodes:
        kept = [i for i in kept if not (n_feat <= i < error_end)]

    if not kept:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No nodes after pruning", ha="center", va="center")
        return fig, ax

    node_info = {idx: _decode_node(graph, idx) for idx in kept}

    # Compact y-axis: map actual layer numbers to plot rows
    active_layers = sorted(set(info[1] for info in node_info.values()))
    layer_to_y = {layer: i for i, layer in enumerate(active_layers)}
    n_y = len(active_layers)

    # Spread multiple features at the same (layer, pos) horizontally
    feat_count: dict = {}
    for idx, (ntype, layer, pos, _) in node_info.items():
        if ntype == "feature":
            feat_count[(layer, pos)] = feat_count.get((layer, pos), 0) + 1
    feat_slot: dict = {}

    positions: dict = {}
    for idx, (ntype, layer, pos, extra) in node_info.items():
        y = layer_to_y[layer]
        if ntype == "feature":
            key = (layer, pos)
            total = feat_count[key]
            slot = feat_slot.get(key, 0)
            feat_slot[key] = slot + 1
            if total > 1:
                x_off = (slot - (total - 1) / 2) / (total - 1) * 0.7
            else:
                x_off = 0.0
            positions[idx] = (pos + x_off, y)
        elif ntype == "logit":
            positions[idx] = (pos + extra * 0.45, y)
        else:
            positions[idx] = (pos, y)

    if figsize is None:
        figsize = (max(8, n_pos * 1.6), max(5, n_y * 1.3))

    fig, ax = plt.subplots(figsize=figsize)

    # --- Edges ---
    adj = graph.adjacency_matrix
    active_edges = edge_mask.nonzero(as_tuple=False)  # shape (E, 2): [dst, src]

    if active_edges.numel() > 0:
        weights = adj[active_edges[:, 0], active_edges[:, 1]]
        max_w = weights.abs().max().item() or 1.0

        for (dst, src), w in zip(active_edges.tolist(), weights.tolist()):
            if src not in positions or dst not in positions:
                continue
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            alpha = float(np.clip(abs(w) / max_w, 0.08, 0.85))
            lw = float(np.clip(2.5 * abs(w) / max_w, 0.4, 2.5))
            color = "#2166ac" if w > 0 else "#d6604d"
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    alpha=alpha,
                    lw=lw,
                    connectionstyle="arc3,rad=0.07",
                ),
                zorder=1,
            )

    # --- Nodes ---
    for idx, (x, y) in positions.items():
        ntype, layer, pos, extra = node_info[idx]
        color = NODE_COLORS[ntype]

        circle = plt.Circle((x, y), NODE_RADIUS, color=color, zorder=3, linewidth=0)
        ax.add_patch(circle)

        if ntype == "feature":
            label = str(extra)
        elif ntype == "token":
            if tokenizer is not None:
                label = tokenizer.decode(graph.input_tokens[pos].item())
                label = label.replace(" ", "·")
            else:
                label = f"T{pos}"
        elif ntype == "logit":
            if tokenizer is not None:
                label = tokenizer.decode(graph.logit_tokens[extra].item())
                label = label.replace(" ", "·")
            else:
                label = f"L{extra}"
        else:
            label = "ε"

        ax.text(
            x, y - NODE_RADIUS - 0.04, label,
            ha="center", va="top", fontsize=7, zorder=4, clip_on=True,
        )

    # --- Axes ---
    ax.set_yticks(list(range(n_y)))
    ylabels = []
    for layer in active_layers:
        if layer == -1:
            ylabels.append("Embed")
        elif layer == n_layers:
            ylabels.append("Logit")
        else:
            ylabels.append(f"Layer {layer}")
    ax.set_yticklabels(ylabels)

    ax.set_xticks(list(range(n_pos)))
    if tokenizer is not None:
        xlabels = [tokenizer.decode(t.item()) for t in graph.input_tokens]
    else:
        xlabels = [str(i) for i in range(n_pos)]
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)

    ax.set_xlim(-0.7, n_pos - 0.3)
    ax.set_ylim(-0.7, n_y - 0.3)
    ax.set_xlabel("Token position")
    ax.grid(True, alpha=0.15, zorder=0)
    ax.set_axisbelow(True)

    # --- Legend ---
    patches = [
        mpatches.Patch(color=NODE_COLORS["token"], label="Token"),
        mpatches.Patch(color=NODE_COLORS["feature"], label="Feature"),
        mpatches.Patch(color=NODE_COLORS["logit"], label="Logit"),
    ]
    if show_error_nodes:
        patches.append(mpatches.Patch(color=NODE_COLORS["error"], label="Error"))
    patches += [
        mpatches.Patch(color="#2166ac", label="Positive edge"),
        mpatches.Patch(color="#d6604d", label="Negative edge"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)

    ax.set_title(title or f'Circuit: "{graph.input_string}"')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


# --- Convenience: load from disk and draw in one call ---

def draw_circuit_from_pt(
    pt_path: str,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    tokenizer=None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Load a .pt graph, prune it, and draw it."""
    graph = Graph.from_pt(pt_path)
    node_mask, edge_mask, cumulative_scores = prune_graph(
        graph, node_threshold=node_threshold, edge_threshold=edge_threshold
    )
    return draw_circuit(graph, node_mask, edge_mask, cumulative_scores, tokenizer=tokenizer, **kwargs)
