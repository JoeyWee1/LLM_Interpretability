import numpy as np
import torch
import plotly.graph_objects as go

from circuit_tracer.graph import Graph


NODE_COLORS = {
    "feature": "#4C72B0",
    "error":   "#DD8452",
    "token":   "#55A868",
    "logit":   "#C44E52",
}


def _decode_node(graph: Graph, idx: int):
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
        return "logit", graph.cfg.n_layers, n_pos - 1, idx - token_end


def build_plotly_circuit(
    graph: Graph,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    cumulative_scores: torch.Tensor,
    tokenizer=None,
    show_error_nodes: bool = False,
) -> tuple[go.Figure, list[dict]]:
    """Build an interactive Plotly circuit figure.

    Returns
    -------
    fig : go.Figure
    node_metadata : list[dict]
        One entry per point in the single node scatter trace.
        Contains type, layer, pos, feat_idx, scan — everything needed
        to fetch feature data on click.
    """
    n_feat = len(graph.selected_features)
    n_layers = graph.cfg.n_layers
    n_pos = graph.n_pos
    error_end = n_feat + n_layers * n_pos

    kept = node_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(kept, int):
        kept = [kept]
    if not show_error_nodes:
        kept = [i for i in kept if not (n_feat <= i < error_end)]

    node_info = {idx: _decode_node(graph, idx) for idx in kept}

    # Compact y-axis
    active_layers = sorted(set(info[1] for info in node_info.values()))
    layer_to_y = {layer: i for i, layer in enumerate(active_layers)}

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
            x_off = (slot - (total - 1) / 2) / max(total - 1, 1) * 0.7 if total > 1 else 0.0
            positions[idx] = (pos + x_off, y)
        elif ntype == "logit":
            positions[idx] = (pos + extra * 0.45, y)
        else:
            positions[idx] = (pos, y)

    traces = []

    # --- Edge traces (not selectable) ---
    adj = graph.adjacency_matrix
    active_edges = edge_mask.nonzero(as_tuple=False)

    if active_edges.numel() > 0:
        weights = adj[active_edges[:, 0], active_edges[:, 1]]
        pos_x, pos_y, neg_x, neg_y = [], [], [], []
        for (dst, src), w in zip(active_edges.tolist(), weights.tolist()):
            if src not in positions or dst not in positions:
                continue
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            if w > 0:
                pos_x += [x0, x1, None]
                pos_y += [y0, y1, None]
            else:
                neg_x += [x0, x1, None]
                neg_y += [y0, y1, None]

        if pos_x:
            traces.append(go.Scatter(
                x=pos_x, y=pos_y, mode="lines",
                line=dict(color="rgba(33,102,172,0.25)", width=1),
                hoverinfo="skip", showlegend=True, name="Positive edge",
            ))
        if neg_x:
            traces.append(go.Scatter(
                x=neg_x, y=neg_y, mode="lines",
                line=dict(color="rgba(214,96,77,0.25)", width=1),
                hoverinfo="skip", showlegend=True, name="Negative edge",
            ))

    # --- Single node trace (all clickable nodes together) ---
    # Order determines evt.index on click, so we store it as node_metadata.
    node_metadata: list[dict] = []
    nx, ny, ncolors, ntext, nhover = [], [], [], [], []

    for idx in kept:
        ntype, layer, pos, extra = node_info[idx]
        x, y = positions[idx]

        if ntype == "feature":
            act = graph.activation_values[graph.selected_features[idx]].item()
            label = f"F{extra}"
            hover = (
                f"<b>Feature {extra}</b><br>"
                f"Layer {layer} · Position {pos}<br>"
                f"Activation: {act:.3f}"
            )
            if tokenizer:
                tok = tokenizer.decode(graph.input_tokens[pos].item())
                hover += f"<br>Token: <i>{tok}</i>"
            meta = {"type": "feature", "layer": layer, "pos": pos, "feat_idx": extra,
                    "scan": graph.scan}

        elif ntype == "token":
            tok = tokenizer.decode(graph.input_tokens[pos].item()) if tokenizer else f"T{pos}"
            label = tok.replace(" ", "·")
            hover = f"<b>Token</b>: {tok}<br>Position {pos}"
            meta = {"type": "token", "layer": layer, "pos": pos, "feat_idx": None,
                    "scan": None}

        elif ntype == "logit":
            tok = tokenizer.decode(graph.logit_tokens[extra].item()) if tokenizer else f"L{extra}"
            p = graph.logit_probabilities[extra].item()
            label = f"{tok.replace(' ', '·')} ({p:.2f})"
            hover = f"<b>Logit</b>: {tok}<br>P = {p:.3f}"
            meta = {"type": "logit", "layer": layer, "pos": pos, "feat_idx": extra,
                    "logit_token": tok, "logit_prob": p, "scan": None}

        else:  # error
            label = "ε"
            hover = f"<b>Error</b><br>Layer {layer} · Position {pos}"
            meta = {"type": "error", "layer": layer, "pos": pos, "feat_idx": None,
                    "scan": None}

        nx.append(x)
        ny.append(y)
        ncolors.append(NODE_COLORS[ntype])
        ntext.append(label)
        nhover.append(hover)
        node_metadata.append(meta)

    traces.append(go.Scatter(
        x=nx, y=ny,
        mode="markers+text",
        marker=dict(color=ncolors, size=14, line=dict(width=1.5, color="white")),
        text=ntext,
        textposition="bottom center",
        textfont=dict(size=9),
        hovertext=nhover,
        hoverinfo="text",
        showlegend=False,
        name="nodes",
    ))

    # --- Legend dummy traces for node colours ---
    for ntype, color in NODE_COLORS.items():
        if ntype == "error" and not show_error_nodes:
            continue
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=10),
            name=ntype.capitalize(), showlegend=True,
        ))

    # --- Axis labels ---
    ytick_vals = list(range(len(active_layers)))
    ytick_text = []
    for layer in active_layers:
        if layer == -1:
            ytick_text.append("Embed")
        elif layer == graph.cfg.n_layers:
            ytick_text.append("Logit")
        else:
            ytick_text.append(f"Layer {layer}")

    xtick_vals = list(range(n_pos))
    if tokenizer:
        xtick_text = [tokenizer.decode(t.item()) for t in graph.input_tokens]
    else:
        xtick_text = [str(i) for i in range(n_pos)]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Circuit: "{graph.input_string}"',
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            tickmode="array", tickvals=xtick_vals, ticktext=xtick_text,
            gridcolor="#eeeeee", zeroline=False,
        ),
        yaxis=dict(
            tickmode="array", tickvals=ytick_vals, ticktext=ytick_text,
            gridcolor="#eeeeee", zeroline=False,
        ),
        margin=dict(l=80, r=20, t=80, b=60),
    )

    return fig, node_metadata
