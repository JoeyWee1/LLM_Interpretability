import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "plt"))
from draw_circuit import _decode_node
from circuit_tracer.graph import Graph, prune_graph

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

DATA_DIR = Path(os.environ.get("PT_DATA_DIR", "/home/zyw26/rds/graphs")).resolve()
if not DATA_DIR.is_dir():
    raise RuntimeError(f"PT_DATA_DIR not found: {DATA_DIR}")

GRAPHS: dict[str, nx.DiGraph] = {}


def build_nx_graph(graph: Graph, node_threshold=0.8, edge_threshold=0.98) -> nx.DiGraph:
    node_mask, edge_mask, cumulative_scores = prune_graph(
        graph, node_threshold=node_threshold, edge_threshold=edge_threshold
    )

    G = nx.DiGraph()
    kept = node_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    active_layers = sorted(set(_decode_node(graph, i)[1] for i in kept))
    layer_to_y    = {l: i for i, l in enumerate(active_layers)}

    for i in kept:
        ntype, layer, pos, extra = _decode_node(graph, i)
        G.add_node(str(i),
            ntype=ntype,
            layer=layer,
            pos=pos,
            extra=extra,
            x=pos,
            y=layer_to_y[layer],
            score=float(cumulative_scores[i]),
            label=str(extra) if extra is not None else f"{ntype}:{pos}",
        )

    adj = graph.adjacency_matrix
    for dst, src in edge_mask.nonzero(as_tuple=False).tolist():
        if node_mask[dst] and node_mask[src]:
            G.add_edge(str(src), str(dst), weight=float(adj[dst, src]))

    return G


@app.get("/files")
def list_files():
    files = []
    for p in sorted(DATA_DIR.rglob("*.pt")):
        rel = p.relative_to(DATA_DIR)
        files.append({
            "name":     str(rel),
            "size_mb":  round(p.stat().st_size / (1024 * 1024), 1),
            "loaded":   str(rel) in GRAPHS,
        })
    return files


@app.post("/load/{filename:path}")
def load_file(filename: str):
    target = (DATA_DIR / filename).resolve()
    if not target.is_file() or DATA_DIR not in target.parents:
        raise HTTPException(404, "File not found in data directory")
    if target.suffix != ".pt":
        raise HTTPException(400, "Not a .pt file")

    if filename in GRAPHS:
        G = GRAPHS[filename]
        return {"filename": filename, "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(), "cached": True}

    graph = Graph.from_pt(target)
    G = build_nx_graph(graph)
    GRAPHS[filename] = G
    return {"filename": filename, "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(), "cached": False}


@app.delete("/load/{filename:path}")
def unload_file(filename: str):
    GRAPHS.pop(filename, None)
    return {"unloaded": filename}


@app.get("/graph/{filename:path}")
def get_graph(filename: str):
    G = _get(filename)
    NODE_COLORS = {
        "feature": "#4C72B0", "token": "#55A868",
        "logit": "#C44E52",   "error": "#DD8452",
    }
    nodes = [{
        "data": {
            "id":    n,
            "label": G.nodes[n].get("label", n),
            "ntype": G.nodes[n].get("ntype", "feature"),
            "color": NODE_COLORS.get(G.nodes[n].get("ntype", ""), "#999"),
            "score": G.nodes[n].get("score", 0),
            "layer": G.nodes[n].get("layer"),
            "pos":   G.nodes[n].get("pos"),
        },
        "position": {
            "x": G.nodes[n].get("x", 0) * 120,
            "y": -G.nodes[n].get("y", 0) * 100,
        },
    } for n in G.nodes()]

    edges = [{
        "data": {
            "id":     f"{u}->{v}",
            "source": u,
            "target": v,
            "weight": G[u][v].get("weight", 0),
            "color":  "#2166ac" if G[u][v].get("weight", 0) > 0 else "#d6604d",
        }
    } for u, v in G.edges()]

    return {"nodes": nodes, "edges": edges}


@app.get("/graph/{filename:path}/node/{node_id}")
def node_details(filename: str, node_id: str):
    G = _get(filename)
    if node_id not in G:
        raise HTTPException(404, "Node not found")
    return {
        "id":         node_id,
        "attributes": dict(G.nodes[node_id]),
        "in_degree":  G.in_degree(node_id),
        "out_degree": G.out_degree(node_id),
        "neighbors":  list(G.neighbors(node_id))[:50],
    }


def _get(filename: str) -> nx.DiGraph:
    if filename not in GRAPHS:
        raise HTTPException(404, "File not loaded — POST /load/{filename} first")
    return GRAPHS[filename]
