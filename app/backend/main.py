from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import networkx as nx
import shutil
import os
from pathlib import Path

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

DATA_DIR = Path(os.environ.get("PT_DATA_DIR", "/home/zyw26/rds/graphs")).resolve()
if not DATA_DIR.is_dir():
    raise RuntimeError(f"PT_DATA_DIR not found: {DATA_DIR}")

# Store dictionary of graphs {filename: nx.Graph}
GRAPHS = {}

# Endpoints for using the files
@app.get("/files")
def list_files():
    """List all available .pt files for the picker"""
    files = [] # list of dictionaries
    for p in sorted(DATA_DIR.rglob(".pt")):
        rel = p.relative_to(DATA_DIR)
    files.append({
            "name": str(rel),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
            "loaded": str(rel) in GRAPHS,
    })
    return files

@app.post("/load/{filename:path}")
def load_file(filename: str):
    """Load .pt from DATA_DIR using relative path"""

    # Safety checks
    target = (DATA_DIR / filename).resolve()
    if not target.is_file() or DATA_DIR not in target.parents:
        raise HTTPException(404, "File not found in data directory")
    if target.suffix != ".pt":
        raise HTTPException(400, "Not a .pt file")
    
    # 