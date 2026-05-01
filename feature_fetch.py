import requests


def cantor_pair(layer: int, feat_idx: int) -> int:
    x, y = layer, feat_idx
    return (x + y) * (x + y + 1) // 2 + y


def fetch_feature(scan: str, layer: int, feat_idx: int, timeout: int = 10) -> dict | None:
    """Fetch feature activation data from the transcoder HuggingFace repo.

    Returns the JSON dict, or None on failure.
    """
    feature_id = cantor_pair(layer, feat_idx)
    url = f"https://huggingface.co/{scan}/resolve/main/features/{feature_id}.json"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def format_feature_html(data: dict) -> str:
    """Render feature activation examples as HTML with orange token highlighting."""
    if data is None:
        return "<p><i>Could not load feature data (no internet access or feature not found).</i></p>"

    html_parts = []

    freq = data.get("activation_frequency", None)
    top_logits = data.get("top_logits", [])
    bottom_logits = data.get("bottom_logits", [])

    if freq is not None:
        html_parts.append(f"<p><b>Activation frequency:</b> {freq:.4%}</p>")
    if top_logits:
        html_parts.append(f"<p><b>Top logits:</b> {', '.join(top_logits[:10])}</p>")
    if bottom_logits:
        html_parts.append(f"<p><b>Bottom logits:</b> {', '.join(bottom_logits[:5])}</p>")

    for quantile in data.get("examples_quantiles", []):
        qname = quantile.get("quantile_name", "")
        examples = quantile.get("examples", [])
        if not examples:
            continue

        html_parts.append(f"<h4 style='margin:12px 0 4px'>{qname}</h4>")

        for ex in examples[:5]:
            tokens = ex.get("tokens", [])
            acts = ex.get("tokens_acts_list", [])
            peak = ex.get("train_token_ind", 0)

            if not tokens:
                continue

            act_max = max(acts) if acts else 1.0

            spans = []
            for i, (tok, act) in enumerate(zip(tokens, acts)):
                intensity = act / act_max if act_max > 0 else 0
                alpha = intensity * 0.85
                border = "2px solid #e65c00" if i == peak else "none"
                bg = f"rgba(255, 140, 0, {alpha:.2f})"
                tok_escaped = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                spans.append(
                    f'<span style="background:{bg}; border-bottom:{border}; '
                    f'padding:1px 2px; border-radius:2px; white-space:pre;">{tok_escaped}</span>'
                )

            html_parts.append(
                f'<div style="margin:4px 0; font-family:monospace; font-size:13px; '
                f'line-height:1.8; word-wrap:break-word;">{"".join(spans)}</div>'
            )

    return "\n".join(html_parts) if html_parts else "<p><i>No data.</i></p>"
