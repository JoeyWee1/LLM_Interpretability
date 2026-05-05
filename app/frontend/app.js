const API = 'http://localhost:8000';

let cy          = null;
let currentFile = null;
let fileList    = [];
let layerData   = []; // [{label, graphY}]

const TYPE_COLORS = {
  feature: 'var(--col-feature)',
  token:   'var(--col-token)',
  logit:   'var(--col-logit)',
  error:   'var(--col-error)',
};

// ── File list ─────────────────────────────────────────────────────────────

async function refreshFiles() {
  try {
    fileList = await apiFetch('/files');
    const sel = document.getElementById('file-select');
    const prev = sel.value;
    sel.innerHTML = fileList.length === 0
      ? '<option value="">— no .pt files found —</option>'
      : fileList.map(f =>
          `<option value="${escAttr(f.name)}">${f.loaded ? '● ' : '  '}${f.name}</option>`
        ).join('');
    if (prev) sel.value = prev;
    updateFileMeta();
  } catch {
    setStatus('Cannot reach backend', true);
  }
}

function updateFileMeta() {
  const name = document.getElementById('file-select').value;
  const f    = fileList.find(x => x.name === name);
  document.getElementById('file-meta').textContent = f ? `${f.size_mb} MB` : '';
}

document.getElementById('file-select').addEventListener('change', updateFileMeta);
document.getElementById('btn-refresh').addEventListener('click', refreshFiles);

// ── Load / Unload ─────────────────────────────────────────────────────────

document.getElementById('btn-load').addEventListener('click', async () => {
  const name = document.getElementById('file-select').value;
  if (!name) return;
  setStatus('Loading…');
  try {
    const meta = await apiFetch(`/load/${encodeURIComponent(name)}`, 'POST');
    currentFile = meta.filename;
    setStatus(
      `${meta.nodes} nodes · ${meta.edges} edges` +
      (meta.cached ? '\n<span style="color:var(--accent)">cached</span>' : '')
    );
    document.getElementById('header-file').textContent   = meta.filename;
    document.getElementById('header-counts').textContent =
      `${meta.nodes}n  ${meta.edges}e`;
    await renderGraph();
    refreshFiles();
  } catch (e) {
    setStatus(e.message, true);
  }
});

document.getElementById('btn-unload').addEventListener('click', async () => {
  if (!currentFile) return;
  await apiFetch(`/load/${encodeURIComponent(currentFile)}`, 'DELETE');
  currentFile = null;
  if (cy) { cy.destroy(); cy = null; }
  clearLayerLabels();
  clearInspector();
  document.getElementById('empty-state').style.display = '';
  document.getElementById('header-file').textContent   = '';
  document.getElementById('header-counts').textContent = '';
  setStatus('No file loaded');
  refreshFiles();
});

// ── Graph render ──────────────────────────────────────────────────────────

async function renderGraph() {
  const data = await apiFetch(`/graph/${encodeURIComponent(currentFile)}`);
  if (cy) cy.destroy();

  document.getElementById('empty-state').style.display = 'none';

  // Work out which layer label goes with which graphY
  let maxLayer = -Infinity;
  data.nodes.forEach(n => {
    const l = n.data.layer;
    if (l != null && l > maxLayer) maxLayer = l;
  });

  const seen = new Map();
  data.nodes.forEach(n => {
    const l = n.data.layer;
    if (l == null || seen.has(l)) return;
    const label = l === -1 ? 'Embed' : l === maxLayer ? 'Logit' : `L${l}`;
    seen.set(l, { label, graphY: n.position.y });
  });
  layerData = Array.from(seen.values());

  cy = cytoscape({
    container: document.getElementById('cy'),
    elements:  [...data.nodes, ...data.edges],
    layout:    { name: 'preset' },
    style:     buildStyle(),
    minZoom: 0.08,
    maxZoom: 5,
  });

  buildLayerLabels();
  updateLayerLabels();
  cy.on('viewport', updateLayerLabels);

  cy.on('tap', 'node', async (evt) => {
    highlightNode(evt.target);
    await showInspector(evt.target.id());
  });

  cy.on('tap', (evt) => {
    if (evt.target === cy) { resetHighlight(); clearInspector(); }
  });

  cy.fit(cy.elements(), 80);
}

function buildStyle() {
  return [
    {
      selector: 'node',
      style: {
        'background-color':      'data(color)',
        'width':                 ele => nodeSize(ele),
        'height':                ele => nodeSize(ele),
        'label':                 'data(label)',
        'font-family':           'JetBrains Mono, monospace',
        'font-size':             '8px',
        'font-weight':           '400',
        'color':                 '#8b949e',
        'text-valign':           'bottom',
        'text-margin-y':         '3px',
        'text-background-color': '#0d1117',
        'text-background-opacity': 0.75,
        'text-background-padding': '1px',
        'border-width':          0,
        'transition-property':   'opacity, border-width',
        'transition-duration':   '80ms',
      }
    },
    {
      selector: 'node.selected',
      style: {
        'border-width': 2,
        'border-color': '#58a6ff',
        'color':        '#e6edf3',
      }
    },
    {
      selector: 'node.dim',
      style: { 'opacity': 0.15 }
    },
    {
      selector: 'edge',
      style: {
        'width':               ele => Math.max(0.5, Math.min(3.5, Math.abs(ele.data('weight') || 0) * 4)),
        'line-color':          'data(color)',
        'target-arrow-color':  'data(color)',
        'target-arrow-shape':  'triangle',
        'arrow-scale':         0.55,
        'curve-style':         'bezier',
        'opacity':             0.45,
        'transition-property': 'opacity',
        'transition-duration': '80ms',
      }
    },
    {
      selector: 'edge.lit',
      style: { 'opacity': 0.9 }
    },
    {
      selector: 'edge.dim',
      style: { 'opacity': 0.04 }
    },
  ];
}

function nodeSize(ele) {
  // cumulative score: lower = more important; map 0→26, 1→14
  const s = ele.data('score') ?? 0.5;
  return Math.round(26 - s * 12);
}

// ── Highlight ─────────────────────────────────────────────────────────────

function highlightNode(node) {
  cy.elements().removeClass('selected lit dim');
  const edges   = node.connectedEdges();
  const touched = edges.connectedNodes();
  cy.elements().not(node).not(edges).not(touched).addClass('dim');
  edges.addClass('lit');
  node.addClass('selected');
}

function resetHighlight() {
  cy.elements().removeClass('selected lit dim');
}

// ── Layer labels ──────────────────────────────────────────────────────────

function buildLayerLabels() {
  const container = document.getElementById('layer-labels');
  container.innerHTML = '';
  layerData.forEach(({ label }) => {
    const el = document.createElement('div');
    el.className = 'layer-label';
    el.id = `ll-${label}`;
    el.textContent = label;
    container.appendChild(el);
  });
}

function updateLayerLabels() {
  if (!cy) return;
  const pan  = cy.pan();
  const zoom = cy.zoom();
  const h    = document.getElementById('canvas-wrap').clientHeight;

  layerData.forEach(({ label, graphY }) => {
    const el = document.getElementById(`ll-${label}`);
    if (!el) return;
    const screenY = graphY * zoom + pan.y;
    el.style.top     = `${screenY}px`;
    el.style.opacity = screenY > 12 && screenY < h - 12 ? '1' : '0';
  });
}

function clearLayerLabels() {
  document.getElementById('layer-labels').innerHTML = '';
  layerData = [];
}

// ── Inspector ─────────────────────────────────────────────────────────────

async function showInspector(nodeId) {
  const data = await apiFetch(
    `/graph/${encodeURIComponent(currentFile)}/node/${encodeURIComponent(nodeId)}`
  );
  const a = data.attributes;

  const color = TYPE_COLORS[a.ntype] || 'var(--text)';

  const rows = [
    ['type',     a.ntype     ?? '—'],
    ['layer',    a.layer === -1 ? 'embed' : (a.layer ?? '—')],
    ['position', a.pos       ?? '—'],
    ['feat idx', a.extra     ?? '—'],
    ['score',    a.score     != null ? a.score.toFixed(5) : '—'],
    ['in',       data.in_degree],
    ['out',      data.out_degree],
  ];

  const inNodes  = cy ? cy.getElementById(nodeId).incomers('node')  : cy.$();
  const outNodes = cy ? cy.getElementById(nodeId).outgoers('node') : cy.$();

  const chips = (nodes) => {
    if (!nodes.length) return '<span class="none-text">—</span>';
    return nodes.map(n => {
      const ntype = n.data('ntype') || 'feature';
      const dot   = `<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${TYPE_COLORS[ntype]||'#999'};margin-right:3px;vertical-align:middle"></span>`;
      return `<div class="chip" onclick="selectNode('${n.id()}')">${dot}${n.data('label') || n.id()}</div>`;
    }).join('');
  };

  document.getElementById('inspector-empty').style.display = 'none';
  const content = document.getElementById('inspector-content');
  content.style.display = 'flex';
  content.innerHTML = `
    <div class="section">
      <div style="margin-bottom:8px">
        <span class="ntype-badge" style="color:${color};border-color:${color}">
          ${(a.ntype || '?').toUpperCase()}
        </span>
      </div>
      ${rows.map(([k, v]) => `
        <div class="irow">
          <span class="ikey">${k}</span>
          <span class="ival">${v}</span>
        </div>`).join('')}
    </div>
    <div class="section">
      <div class="section-label">Inputs (${inNodes.length})</div>
      <div class="chips">${chips(inNodes)}</div>
    </div>
    <div class="section">
      <div class="section-label">Outputs (${outNodes.length})</div>
      <div class="chips">${chips(outNodes)}</div>
    </div>
  `;
}

function clearInspector() {
  document.getElementById('inspector-empty').style.display = '';
  const c = document.getElementById('inspector-content');
  c.style.display = 'none';
  c.innerHTML = '';
}

function selectNode(id) {
  if (!cy) return;
  const node = cy.getElementById(id);
  if (!node.length) return;
  cy.animate({ center: { eles: node } }, { duration: 180 });
  highlightNode(node);
  showInspector(id);
}

// ── Utilities ─────────────────────────────────────────────────────────────

async function apiFetch(path, method = 'GET') {
  const res = await fetch(`${API}${path}`, { method });
  if (!res.ok) {
    const txt = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${txt}`);
  }
  return res.json();
}

function setStatus(html, isError = false) {
  const el = document.getElementById('status-text');
  el.innerHTML = html;
  el.style.color = isError ? 'var(--col-logit)' : 'var(--text-dim)';
}

function escAttr(s) {
  return s.replace(/"/g, '&quot;');
}

// ── Boot ──────────────────────────────────────────────────────────────────

refreshFiles();
