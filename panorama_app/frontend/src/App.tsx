import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import OpenSeadragon from "openseadragon";
import { Download, Eraser, Hand, Paintbrush, Play, RefreshCw, RotateCcw, Square, Upload } from "lucide-react";
import { API_BASE, ProjectInfo, ClassificationResult, api } from "./api";
import "./styles.css";

type Tool = "pan" | "add" | "erase";

function StatsPanel({ project }: { project: ProjectInfo | null }) {
  const stats = project?.stats;
  return (
    <section className="panel">
      <h2>Mask stats</h2>
      {!stats ? (
        <p className="muted">No stats yet</p>
      ) : (
        <dl className="stats">
          <dt>Fill</dt><dd>{stats.fill_percent.toFixed(2)}%</dd>
          <dt>Mask pixels</dt><dd>{stats.mask_pixels.toLocaleString()}</dd>
          <dt>Total pixels</dt><dd>{stats.total_pixels.toLocaleString()}</dd>
          <dt>Components</dt><dd>{stats.component_count}</dd>
          <dt>Largest</dt><dd>{stats.largest_component_pixels.toLocaleString()}</dd>
          <dt>BBox</dt><dd>{stats.largest_component_bbox?.join(", ") ?? "none"}</dd>
        </dl>
      )}
    </section>
  );
}

function ClassificationPanel({ projectId }: { projectId: string | null }) {
  const [result, setResult] = useState<ClassificationResult | null>(null);
  useEffect(() => {
    if (!projectId) return;
    api.classification(projectId).then(setResult).catch(() => setResult(null));
  }, [projectId]);
  return (
    <section className="panel">
      <h2>Classification</h2>
      {!result ? <p className="muted">No project selected</p> : (
        <dl className="stats">
          <dt>Class</dt><dd>{result.class_name}</dd>
          <dt>Confidence</dt><dd>{result.confidence ?? "placeholder"}</dd>
          <dt>Version</dt><dd>{result.model_version}</dd>
        </dl>
      )}
    </section>
  );
}

function Viewer({
  project,
  tool,
  brushSize,
  maskOpacity,
  maskRevision,
  onEdited,
}: {
  project: ProjectInfo | null;
  tool: Tool;
  brushSize: number;
  maskOpacity: number;
  maskRevision: number;
  onEdited: (project: ProjectInfo) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const pointsRef = useRef<number[][]>([]);

  const makeTileSource = (id: string, kind: "image" | "mask", width: number, height: number, revision = 0) => {
    const maxLevel = Math.max(0, Math.ceil(Math.log2(Math.max(width, height) / 256)));
    return {
      width,
      height,
      tileSize: 256,
      minLevel: 0,
      maxLevel,
      getTileUrl: (level: number, x: number, y: number) =>
        `${API_BASE}/api/projects/${id}/tiles/${kind}/${level}/${x}/${y}.${kind === "image" ? "jpg" : "png"}?v=${revision}`,
    };
  };

  useEffect(() => {
    if (!project || !containerRef.current) return;
    viewerRef.current?.destroy();
    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/5.0.1/images/",
      tileSources: makeTileSource(project.id, "image", project.image.width, project.image.height),
      showNavigator: true,
      gestureSettingsMouse: { clickToZoom: false, dragToPan: true },
    });
    viewerRef.current = viewer;
    return () => viewer.destroy();
  }, [project?.id, project?.status]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !project || project.status !== "ready") return;
    if (viewer.world.getItemCount() > 1) viewer.world.removeItem(viewer.world.getItemAt(1));
    viewer.addTiledImage({
      tileSource: makeTileSource(project.id, "mask", project.image.width, project.image.height, maskRevision),
      opacity: maskOpacity,
    });
  }, [maskRevision, project?.id, project?.status]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.world.getItemCount() < 2) return;
    viewer.world.getItemAt(1).setOpacity(maskOpacity);
  }, [maskOpacity]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !project) return;
    const activeViewer = viewer;
    activeViewer.setMouseNavEnabled(tool === "pan");
    if (tool === "pan") return;

    const tracker = new OpenSeadragon.MouseTracker({
      element: viewer.canvas,
      pressHandler: (event: any) => {
        event.preventDefaultAction = true;
        pointsRef.current = [viewportPointToImage(event.position)];
      },
      dragHandler: (event: any) => {
        event.preventDefaultAction = true;
        pointsRef.current.push(viewportPointToImage(event.position));
      },
      releaseHandler: async () => {
        if (pointsRef.current.length === 0) return;
        const updated = await api.edit(project.id, tool, pointsRef.current, brushSize);
        pointsRef.current = [];
        onEdited(updated);
        refreshMaskLayer(project.id);
      },
    });

    function viewportPointToImage(position: OpenSeadragon.Point): number[] {
      const viewport = activeViewer.viewport.pointFromPixel(position);
      const image = activeViewer.viewport.viewportToImageCoordinates(viewport);
      return [Math.round(image.x), Math.round(image.y)];
    }

    function refreshMaskLayer(id: string) {
      if (!project) return;
      if (activeViewer.world.getItemCount() > 1) activeViewer.world.removeItem(activeViewer.world.getItemAt(1));
      activeViewer.addTiledImage({
        tileSource: makeTileSource(id, "mask", project.image.width, project.image.height, Date.now()),
        opacity: maskOpacity,
      });
    }

    tracker.setTracking(true);
    return () => tracker.destroy();
  }, [project?.id, tool, brushSize, maskOpacity]);

  return <div ref={containerRef} className="viewer" />;
}

function App() {
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<ProjectInfo | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [maskOpacity, setMaskOpacity] = useState(0.45);
  const [brushSize, setBrushSize] = useState(24);
  const [tool, setTool] = useState<Tool>("pan");
  const [busy, setBusy] = useState(false);
  const [maskRevision, setMaskRevision] = useState(0);
  const thresholdTimerRef = useRef<number | null>(null);

  const refreshProjects = async () => setProjects(await api.projects());

  useEffect(() => { refreshProjects(); }, []);
  useEffect(() => {
    if (!selectedId) return;
    const load = async () => {
      const p = await api.project(selectedId);
      setSelected(p);
      setThreshold(p.threshold);
    };
    load();
    const timer = setInterval(load, 2500);
    return () => clearInterval(timer);
  }, [selectedId]);

  const upload = async (file: File) => {
    setBusy(true);
    try {
      const p = await api.upload(file);
      await refreshProjects();
      setSelectedId(p.id);
    } finally {
      setBusy(false);
    }
  };

  const runInference = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      const p = await api.infer(selected.id);
      setSelected(p);
      await refreshProjects();
    } finally {
      setBusy(false);
    }
  };

  const cancelInference = async () => {
    if (!selected) return;
    const p = await api.cancel(selected.id);
    setSelected(p);
    await refreshProjects();
  };

  const commitThreshold = async (value: number) => {
    if (!selected || selected.status !== "ready") return;
    if (thresholdTimerRef.current !== null) {
      window.clearTimeout(thresholdTimerRef.current);
      thresholdTimerRef.current = null;
    }
    const p = await api.threshold(selected.id, value);
    setSelected(p);
    setMaskRevision((x) => x + 1);
  };

  const updateThreshold = (value: number) => {
    setThreshold(value);
    if (!selected || selected.status !== "ready") return;
    if (thresholdTimerRef.current !== null) {
      window.clearTimeout(thresholdTimerRef.current);
    }
    thresholdTimerRef.current = window.setTimeout(() => {
      commitThreshold(value).catch(console.error);
      thresholdTimerRef.current = null;
    }, 450);
  };

  const resetAll = async () => {
    if (!selected || selected.status !== "ready") return;
    const p = await api.reset(selected.id);
    setSelected(p);
    setThreshold(p.threshold);
    setMaskRevision((x) => x + 1);
    await refreshProjects();
  };

  const exportZip = async () => {
    if (!selected) return;
    const res = await fetch(api.exportUrl(selected.id), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "zip" }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selected.id}_export.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="brand">Talc Mask Editor</div>
        <label className="upload">
          <Upload size={18} />
          Upload image
          <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        </label>
          <button disabled={!selected || busy} onClick={runInference}><Play size={16} /> Run inference</button>
        <button disabled={!selected || selected.status !== "running"} onClick={cancelInference}><Square size={16} /> Cancel</button>
        <button onClick={refreshProjects}><RefreshCw size={16} /> Refresh</button>
        <div className="projectList">
          {projects.map((p) => (
            <button key={p.id} className={p.id === selectedId ? "selected" : ""} onClick={() => setSelectedId(p.id)}>
              <span>{p.image.filename}</span>
              <small>{p.status} · {p.image.width}x{p.image.height}</small>
            </button>
          ))}
        </div>
      </aside>

      <section className="workspace">
        <div className="toolbar">
          <button className={tool === "pan" ? "active" : ""} onClick={() => setTool("pan")}><Hand size={16} /> Pan</button>
          <button className={tool === "add" ? "active" : ""} disabled={selected?.status !== "ready"} onClick={() => setTool("add")}><Paintbrush size={16} /> Add</button>
          <button className={tool === "erase" ? "active" : ""} disabled={selected?.status !== "ready"} onClick={() => setTool("erase")}><Eraser size={16} /> Erase</button>
          <label>Brush <input type="range" min="2" max="128" value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} /> {brushSize}px</label>
          <label>Threshold <input type="range" min="0" max="1" step="0.01" value={threshold} disabled={selected?.status !== "ready"} onChange={(e) => updateThreshold(Number(e.target.value))} onPointerUp={() => commitThreshold(threshold).catch(console.error)} /> {threshold.toFixed(2)}</label>
          <label>Opacity <input type="range" min="0" max="1" step="0.05" value={maskOpacity} onChange={(e) => setMaskOpacity(Number(e.target.value))} /> {maskOpacity.toFixed(2)}</label>
          <button disabled={selected?.status !== "ready"} onClick={resetAll}><RotateCcw size={16} /> Reset all</button>
          <button disabled={!selected} onClick={exportZip}><Download size={16} /> Export</button>
        </div>
        <Viewer
          project={selected}
          tool={tool}
          brushSize={brushSize}
          maskOpacity={maskOpacity}
          maskRevision={maskRevision}
          onEdited={(project) => {
            setSelected(project);
            setMaskRevision((x) => x + 1);
          }}
        />
      </section>

      <aside className="inspector">
        <section className="panel">
          <h2>Project</h2>
          {selected ? (
            <dl className="stats">
              <dt>Status</dt><dd>{selected.status}</dd>
              <dt>ID</dt><dd>{selected.id}</dd>
              <dt>Progress</dt><dd>{typeof selected.inference_progress?.percent === "number" ? `${Number(selected.inference_progress.percent).toFixed(1)}%` : "none"}</dd>
              <dt>Error</dt><dd>{selected.error ?? "none"}</dd>
            </dl>
          ) : <p className="muted">Select a project</p>}
        </section>
        <StatsPanel project={selected} />
        <ClassificationPanel projectId={selected?.id ?? null} />
      </aside>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
