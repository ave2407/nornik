import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import OpenSeadragon from "openseadragon";
import { Download, Eraser, Hand, Paintbrush, Play, RotateCcw, Square, Trash2, Upload } from "lucide-react";
import { API_BASE, ProjectInfo, ClassificationResult, api } from "./api";
import Home from "./Home";
import "./styles.css";

type Page = "home" | "tool";

function SiteNav({ page, onNavigate }: { page: Page; onNavigate: (page: Page) => void }) {
  return (
    <nav className="siteNav">
      <div className="siteNavLinks">
        <button className={page === "home" ? "active" : ""} onClick={() => onNavigate("home")}>
          Главная
        </button>
        <button className={page === "tool" ? "active" : ""} onClick={() => onNavigate("tool")}>
          Инструмент
        </button>
      </div>
    </nav>
  );
}

type Tool = "pan" | "add" | "erase";
type OverlayMode = "talc" | "phases" | "heatmap";

function overlayKind(mode: OverlayMode): "mask" | "phases" | "heatmap" {
  if (mode === "talc") return "mask";
  return mode;
}

const STATUS_LABELS: Record<string, string> = {
  created: "загружено",
  running: "анализ идёт…",
  ready: "готово",
  failed: "ошибка",
  cancelled: "отменено",
};

function statusLabel(status: string) {
  return STATUS_LABELS[status] ?? status;
}

function pct(value?: number) {
  return typeof value === "number" ? `${value.toFixed(2)}%` : "нет данных";
}

function MiniBar({ label, percent, color }: { label: string; percent: number; color: string }) {
  const clamped = Math.min(100, Math.max(0, percent));
  return (
    <div className="miniBarRow">
      <span className="miniBarLabel">{label}</span>
      <span className="miniBarTrack">
        <span className="miniBarFill" style={{ width: `${clamped}%`, background: color }} />
      </span>
      <span className="miniBarValue">{clamped.toFixed(0)}%</span>
    </div>
  );
}

function StatsPanel({ project }: { project: ProjectInfo | null }) {
  const stats = project?.stats;
  return (
    <section className="panel">
      <h2>Статистика маски талька</h2>
      {!stats ? (
        <p className="muted">Пока нет данных</p>
      ) : (
        <>
          <MiniBar label="Заполнение маски" percent={stats.fill_percent} color="var(--accent)" />
          <div className="statTiles">
            <div className="statTile"><span>Пикселей маски</span><b>{stats.mask_pixels.toLocaleString()}</b></div>
            <div className="statTile"><span>Всего пикселей</span><b>{stats.total_pixels.toLocaleString()}</b></div>
            <div className="statTile"><span>Компонентов</span><b>{stats.component_count}</b></div>
            <div className="statTile"><span>Наибольший фрагмент</span><b>{stats.largest_component_pixels.toLocaleString()}</b></div>
          </div>
          {stats.largest_component_bbox && (
            <p className="muted bboxNote">BBox: {stats.largest_component_bbox.join(", ")}</p>
          )}
        </>
      )}
    </section>
  );
}

function ClassificationPanel({ project }: { project: ProjectInfo | null }) {
  const [result, setResult] = useState<ClassificationResult | null>(null);
  useEffect(() => {
    if (!project || project.status !== "ready") {
      setResult(null);
      return;
    }
    api.classification(project.id).then(setResult).catch(() => setResult(null));
  }, [project?.id, project?.status, project?.updated_at]);
  const stats = result?.phase_stats;
  return (
    <section className="panel">
      <h2>Класс руды</h2>
      {!result ? <p className="muted">Запустите анализ, чтобы получить классификацию</p> : (
        <>
          <div className={`classBadge ${result.class_name}`}>
            <span className="classBadgeName">{result.display_name || result.class_name}</span>
            {result.confidence !== null && <span className="classBadgeConf">{(result.confidence * 100).toFixed(0)}%</span>}
          </div>

          <p className="miniGroupLabel">Вероятности классификатора</p>
          <MiniBar label="Рядовая" percent={(result.model_probs?.ordinary ?? result.probs?.ordinary ?? 0) * 100} color="#18a34a" />
          <MiniBar label="Труднообог." percent={(result.model_probs?.difficult ?? result.probs?.difficult ?? 0) * 100} color="#dc2626" />
          <MiniBar label="Оталькован." percent={(result.model_probs?.talc ?? result.probs?.talc ?? 0) * 100} color="#1455ff" />

          {stats && (
            <>
              <p className="miniGroupLabel">Состав изображения</p>
              <div className="compBar">
                <span style={{ width: `${stats.talc_percent}%`, background: "#1455ff" }} />
                <span style={{ width: `${stats.ordinary_intergrowth_area_percent}%`, background: "#18a34a" }} />
                <span style={{ width: `${stats.thin_intergrowth_area_percent}%`, background: "#dc2626" }} />
                <span style={{ flex: 1, background: "#4b5563" }} />
              </div>
              <div className="legend">
                <span><i className="swatch talc" /> тальк {pct(stats.talc_percent)}</span>
                <span><i className="swatch ordinary" /> крупные срастания {pct(stats.ordinary_intergrowth_area_percent)}</span>
                <span><i className="swatch difficult" /> тонкие срастания {pct(stats.thin_intergrowth_area_percent)}</span>
                <span><i className="swatch gangue" /> нерудная матрица {pct(stats.gangue_percent)}</span>
              </div>
              <p className="muted sulfideNote">Сульфиды (крупные + тонкие вместе): {pct(stats.sulfide_percent)}</p>
              <div className="statTiles">
                <div className="statTile"><span>Мелких компонентов</span><b>{stats.fine_component_count}</b></div>
                <div className="statTile"><span>Крупных компонентов</span><b>{stats.coarse_component_count}</b></div>
              </div>
            </>
          )}

          <p className="reason">{result.decision_reason}</p>
        </>
      )}
    </section>
  );
}

function ExportDialog({
  projects,
  initialId,
  onClose,
  onConfirm,
}: {
  projects: ProjectInfo[];
  initialId: string;
  onClose: () => void;
  onConfirm: (ids: string[]) => void;
}) {
  const [checked, setChecked] = useState<Set<string>>(new Set([initialId]));
  const toggle = (id: string) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  return (
    <div className="modalOverlay" onClick={onClose}>
      <div className="modalPanel" onClick={(e) => e.stopPropagation()}>
        <h3>Экспорт</h3>
        <p className="muted">Выберите один или несколько снимков для архива</p>
        <div className="modalList">
          {projects.map((p) => (
            <label key={p.id} className="modalCheckRow">
              <input type="checkbox" checked={checked.has(p.id)} onChange={() => toggle(p.id)} />
              <span>{p.image.filename}</span>
              <small>{statusLabel(p.status)}</small>
            </label>
          ))}
        </div>
        <div className="modalActions">
          <button onClick={onClose}>Отмена</button>
          <button className="landingCta" disabled={checked.size === 0} onClick={() => onConfirm(Array.from(checked))}>
            <Download size={16} /> Экспортировать ({checked.size})
          </button>
        </div>
      </div>
    </div>
  );
}

function Viewer({
  project,
  tool,
  brushSize,
  maskOpacity,
  maskRevision,
  overlayMode,
  onEdited,
}: {
  project: ProjectInfo | null;
  tool: Tool;
  brushSize: number;
  maskOpacity: number;
  maskRevision: number;
  overlayMode: OverlayMode;
  onEdited: (project: ProjectInfo) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const pointsRef = useRef<number[][]>([]);
  const lastProbeRef = useRef<number | null>(null);
  const probeSeqRef = useRef(0);
  const [brushCursor, setBrushCursor] = useState<{ px: number; py: number; r: number } | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{ px: number; py: number; ix: number; iy: number; prob: number | null } | null>(null);

  const makeTileSource = (id: string, kind: "image" | "mask" | "phases" | "heatmap", width: number, height: number, revision = 0) => {
    const maxLevel = Math.max(0, Math.ceil(Math.log2(Math.max(width, height) / 256)));
    const ext = kind === "image" ? "jpg" : "png";
    return {
      width,
      height,
      tileSize: 256,
      minLevel: 0,
      maxLevel,
      getTileUrl: (level: number, x: number, y: number) =>
        `${API_BASE}/api/projects/${id}/tiles/${kind}/${level}/${x}/${y}.${ext}?v=${revision}`,
    };
  };

  const addOverlayLayer = (viewer: OpenSeadragon.Viewer, projectInfo: ProjectInfo, revision: number) => {
    if (viewer.world.getItemCount() > 1) viewer.world.removeItem(viewer.world.getItemAt(1));
    if (projectInfo.status !== "ready") return;
    viewer.addTiledImage({
      tileSource: makeTileSource(projectInfo.id, overlayKind(overlayMode), projectInfo.image.width, projectInfo.image.height, revision),
      opacity: maskOpacity,
    });
  };

  useEffect(() => {
    if (!project || !containerRef.current) return;
    viewerRef.current?.destroy();
    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/5.0.1/images/",
      tileSources: makeTileSource(project.id, "image", project.image.width, project.image.height),
      showNavigator: true,
      showNavigationControl: false,
      gestureSettingsMouse: { clickToZoom: false, dragToPan: true },
    });
    viewerRef.current = viewer;
    return () => viewer.destroy();
  }, [project?.id, project?.status]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !project) return;
    addOverlayLayer(viewer, project, maskRevision);
  }, [maskRevision, overlayMode, project?.id, project?.status]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.world.getItemCount() < 2) return;
    viewer.world.getItemAt(1).setOpacity(maskOpacity);
  }, [maskOpacity]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !project) {
      setBrushCursor(null);
      return;
    }
    if (tool === "pan") {
      viewer.setMouseNavEnabled(true);
      setBrushCursor(null);
      return;
    }
    const activeViewer = viewer;
    activeViewer.setMouseNavEnabled(false);

    function brushScreenRadius(): number {
      const p0 = activeViewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0));
      const p1 = activeViewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(brushSize, 0));
      return Math.abs(p1.x - p0.x);
    }

    const tracker = new OpenSeadragon.MouseTracker({
      element: viewer.canvas,
      moveHandler: (event: any) => {
        setBrushCursor({ px: event.position.x, py: event.position.y, r: brushScreenRadius() });
      },
      leaveHandler: () => setBrushCursor(null),
      pressHandler: (event: any) => {
        event.preventDefaultAction = true;
        pointsRef.current = [viewportPointToImage(event.position)];
      },
      dragHandler: (event: any) => {
        event.preventDefaultAction = true;
        pointsRef.current.push(viewportPointToImage(event.position));
        setBrushCursor({ px: event.position.x, py: event.position.y, r: brushScreenRadius() });
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
        tileSource: makeTileSource(id, overlayKind(overlayMode), project.image.width, project.image.height, Date.now()),
        opacity: maskOpacity,
      });
    }

    tracker.setTracking(true);
    return () => {
      tracker.destroy();
      activeViewer.setMouseNavEnabled(true);
      setBrushCursor(null);
    };
  }, [project?.id, tool, brushSize, maskOpacity, overlayMode]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !project || project.status !== "ready" || tool !== "pan") {
      setHoverInfo(null);
      return;
    }
    const activeViewer = viewer;
    const hoverTracker = new OpenSeadragon.MouseTracker({
      element: viewer.canvas,
      moveHandler: (event: any) => {
        const viewport = activeViewer.viewport.pointFromPixel(event.position);
        const image = activeViewer.viewport.viewportToImageCoordinates(viewport);
        const ix = Math.round(image.x);
        const iy = Math.round(image.y);
        if (ix < 0 || iy < 0 || ix >= project.image.width || iy >= project.image.height) {
          setHoverInfo(null);
          return;
        }
        setHoverInfo({ px: event.position.x, py: event.position.y, ix, iy, prob: null });
        if (lastProbeRef.current) window.clearTimeout(lastProbeRef.current);
        lastProbeRef.current = window.setTimeout(() => {
          const seq = ++probeSeqRef.current;
          api
            .probabilityAt(project.id, ix, iy)
            .then((res) => {
              if (seq !== probeSeqRef.current) return;
              setHoverInfo((prev) => (prev && prev.ix === ix && prev.iy === iy ? { ...prev, prob: res.probability } : prev));
            })
            .catch(() => {});
        }, 60);
      },
      leaveHandler: () => {
        setHoverInfo(null);
        if (lastProbeRef.current) window.clearTimeout(lastProbeRef.current);
      },
    });
    hoverTracker.setTracking(true);
    return () => hoverTracker.destroy();
  }, [project?.id, project?.status, tool]);

  return (
    <div className="viewerWrap">
      <div ref={containerRef} className="viewer" />
      {hoverInfo && hoverInfo.prob !== null && (
        <div className="probTooltip" style={{ left: hoverInfo.px + 16, top: hoverInfo.py + 16 }}>
          {(hoverInfo.prob * 100).toFixed(0)}%
        </div>
      )}
      {brushCursor && (tool === "add" || tool === "erase") && (
        <div
          className={`brushCursor ${tool}`}
          style={{
            left: brushCursor.px - brushCursor.r,
            top: brushCursor.py - brushCursor.r,
            width: brushCursor.r * 2,
            height: brushCursor.r * 2,
          }}
        />
      )}
    </div>
  );
}

function App() {
  const [page, setPage] = useState<Page>("home");
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<ProjectInfo | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [maskOpacity, setMaskOpacity] = useState(0.45);
  const [brushSize, setBrushSize] = useState(24);
  const [tool, setTool] = useState<Tool>("pan");
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("talc");
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
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

  const startInference = async (id: string) => {
    setBusy(true);
    setErrorMsg(null);
    try {
      const p = await api.infer(id);
      setSelected(p);
      await refreshProjects();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const upload = async (file: File) => {
    setBusy(true);
    setErrorMsg(null);
    try {
      const p = await api.upload(file);
      await refreshProjects();
      setSelectedId(p.id);
      // Загрузка сразу запускает анализ — не нужно жать вторую кнопку.
      await startInference(p.id);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  const runInference = () => {
    if (!selected) return;
    return startInference(selected.id);
  };

  const cancelInference = async () => {
    if (!selected) return;
    const p = await api.cancel(selected.id);
    setSelected(p);
    await refreshProjects();
  };

  const deleteProject = async (id: string) => {
    if (!window.confirm("Удалить этот проект без возможности восстановления?")) return;
    await api.remove(id);
    if (selectedId === id) {
      setSelectedId(null);
      setSelected(null);
    }
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

  const downloadExport = async (ids: string[]) => {
    const res = await fetch(`${API_BASE}/api/export/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_ids: ids }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = ids.length === 1 ? `${ids[0]}_export.zip` : `export_${ids.length}_projects.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const notReady = selected?.status !== "ready";

  return (
    <div className="siteRoot">
      <SiteNav page={page} onNavigate={setPage} />
      <div className={page === "home" ? "siteContent siteContent--scroll" : "siteContent siteContent--fixed"}>
        {page === "home" ? (
          <Home onOpenTool={() => setPage("tool")} />
        ) : (
    <main className="app">
      <aside className="sidebar">
        <label className="upload" title="Загрузите фото шлифа или панораму — анализ запустится автоматически">
          <Upload size={18} />
          Загрузить изображение
          <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        </label>
        {errorMsg && <div className="errorBanner">{errorMsg}</div>}
        {busy && <p className="muted busyHint">Обработка… это может занять некоторое время для больших панорам</p>}
        <button
          className={`sidebarBtn${selected?.status === "running" ? " running" : ""}`}
          disabled={!selected || busy}
          onClick={runInference}
          title="Повторить анализ для выбранного изображения"
        >
          <Play size={16} /> {selected?.status === "running" ? "Анализ идёт…" : selected?.status === "ready" ? "Повторить анализ" : "Запустить анализ"}
        </button>
        <button className="sidebarBtn" disabled={!selected || selected.status !== "running"} onClick={cancelInference}><Square size={16} /> Отменить</button>
        <div className="projectList">
          {projects.length === 0 && <p className="muted">Проектов пока нет — загрузите изображение выше, чтобы начать.</p>}
          {projects.map((p) => (
            <div key={p.id} className={`projectItem${p.id === selectedId ? " selected" : ""}`}>
              <button className="projectItemMain" onClick={() => setSelectedId(p.id)}>
                <span>{p.image.filename}</span>
                <small>{statusLabel(p.status)} · {p.image.width}x{p.image.height}</small>
              </button>
              <button
                className="projectItemDelete"
                data-tip="Удалить проект"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteProject(p.id).catch(console.error);
                }}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <section className="workspace">
        <div className="toolbar">
          <div className="toolbarRow">
            <div className="toolbarRowStart">
              <div className="toolGroup">
                <button className={tool === "pan" ? "active" : ""} onClick={() => setTool("pan")} data-tip="Перемещение и зум"><Hand size={16} /></button>
                <button className={tool === "add" ? "active" : ""} disabled={notReady} onClick={() => setTool("add")} data-tip={notReady ? "Доступно после анализа" : "Кисть: добавить тальк"}><Paintbrush size={16} /></button>
                <button className={tool === "erase" ? "active" : ""} disabled={notReady} onClick={() => setTool("erase")} data-tip={notReady ? "Доступно после анализа" : "Ластик: стереть тальк"}><Eraser size={16} /></button>
              </div>

              <div className="toolGroup">
                <button className={overlayMode === "talc" ? "active" : ""} onClick={() => setOverlayMode("talc")} title="Показать маску талька">Тальк</button>
                <button className={overlayMode === "phases" ? "active" : ""} disabled={notReady} onClick={() => setOverlayMode("phases")} title={notReady ? "Доступно после завершения анализа" : "Показать фазовую разметку (сульфиды/матрица)"}>Фазы</button>
                <button className={overlayMode === "heatmap" ? "active" : ""} disabled={notReady} onClick={() => setOverlayMode("heatmap")} title={notReady ? "Доступно после завершения анализа" : "Показать карту уверенности модели"}>Уверенность</button>
              </div>

              {(tool === "add" || tool === "erase") && (
                <label className="toolSlider" title="Размер кисти для ручной правки маски">
                  Кисть <input type="range" min="2" max="128" value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} /> <b>{brushSize}px</b>
                </label>
              )}
            </div>
            <div className="toolbarRowEnd">
              <label className="toolSlider" title="Прозрачность цветной маски поверх изображения">
                Прозрачность <input type="range" min="0" max="1" step="0.05" value={maskOpacity} onChange={(e) => setMaskOpacity(Number(e.target.value))} /> <b>{maskOpacity.toFixed(2)}</b>
              </label>
            </div>
          </div>

          <div className="toolbarRow">
            <div className="toolbarRowStart">
              {overlayMode === "talc" && (
                <label className="toolSlider" title="Порог вероятности, начиная с которого пиксель считается тальком">
                  Порог <input type="range" min="0" max="1" step="0.01" value={threshold} disabled={notReady} onChange={(e) => updateThreshold(Number(e.target.value))} onPointerUp={() => commitThreshold(threshold).catch(console.error)} /> <b>{threshold.toFixed(2)}</b>
                </label>
              )}
              {selected?.status === "running" && (
                <div className="toolProgress" title="Прогресс анализа">
                  Анализ
                  <span className="toolProgressBar">
                    <span
                      className="toolProgressFill"
                      style={{
                        width: `${typeof selected.inference_progress?.percent === "number" ? selected.inference_progress.percent : 0}%`,
                      }}
                    />
                  </span>
                  <b>{typeof selected.inference_progress?.percent === "number" ? selected.inference_progress.percent.toFixed(0) : 0}%</b>
                </div>
              )}
            </div>
            <div className="toolbarRowEnd">
              <div className="toolGroup">
                <button disabled={notReady} onClick={resetAll} title="Сбросить порог и ручные правки к исходному результату модели"><RotateCcw size={16} /> Сбросить</button>
                <button disabled={!selected} onClick={() => setExportDialogOpen(true)} title="Выбрать снимки и скачать архив"><Download size={16} /> Экспорт</button>
              </div>
            </div>
          </div>
        </div>
        {!selected && (
          <div className="emptyHint">Загрузите изображение слева, чтобы увидеть результат анализа</div>
        )}
        {selected?.status === "ready" && overlayMode === "heatmap" && (
          <div className="heatmapLegend" title="Уверенность модели: от низкой (синий) до высокой (красный)">
            <span>0%</span>
            <span className="heatmapGradient" />
            <span>100%</span>
          </div>
        )}
        <Viewer
          project={selected}
          tool={tool}
          brushSize={brushSize}
          maskOpacity={maskOpacity}
          maskRevision={maskRevision}
          overlayMode={overlayMode}
          onEdited={(project) => {
            setSelected(project);
            setMaskRevision((x) => x + 1);
          }}
        />
      </section>

      <aside className="inspector">
        <section className="panel">
          <h2>Проект</h2>
          {selected ? (
            <>
              <div className="statusRow">
                <span className={`statusDot ${selected.status}`} />
                {statusLabel(selected.status)}
              </div>
              <p className="muted idNote">{selected.id}</p>
              {selected.error && <div className="errorBanner">{selected.error}</div>}
            </>
          ) : <p className="muted">Выберите проект слева</p>}
        </section>
        <ClassificationPanel project={selected} />
        <StatsPanel project={selected} />
      </aside>
      {exportDialogOpen && selected && (
        <ExportDialog
          projects={projects}
          initialId={selected.id}
          onClose={() => setExportDialogOpen(false)}
          onConfirm={(ids) => {
            setExportDialogOpen(false);
            downloadExport(ids).catch(console.error);
          }}
        />
      )}
    </main>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
