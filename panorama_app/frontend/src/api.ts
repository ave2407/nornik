export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:7860";

export type MaskStats = {
  total_pixels: number;
  mask_pixels: number;
  fill_percent: number;
  component_count: number;
  largest_component_pixels: number;
  largest_component_bbox: number[] | null;
};

export type PhaseStats = {
  source_width: number;
  source_height: number;
  analysis_width: number;
  analysis_height: number;
  analysis_scale: number;
  total_pixels: number;
  talc_pixels: number;
  sulfide_pixels: number;
  gangue_pixels: number;
  ordinary_intergrowth_pixels: number;
  thin_intergrowth_pixels: number;
  talc_percent: number;
  sulfide_percent: number;
  gangue_percent: number;
  ordinary_intergrowth_area_percent: number;
  thin_intergrowth_area_percent: number;
  min_component_pixels: number;
  coarse_component_pixels: number;
  sulfide_component_count: number;
  fine_component_count: number;
  coarse_component_count: number;
  largest_sulfide_component_pixels: number;
};

export type ProjectInfo = {
  id: string;
  status: "created" | "running" | "ready" | "failed" | "cancelled";
  threshold: number;
  image: { width: number; height: number; filename: string };
  stats: MaskStats | null;
  error: string | null;
  inference_progress: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

export type ProbabilityPoint = {
  probability: number | null;
};

export type ClassificationResult = {
  project_id: string;
  class_name: string;
  display_name: string;
  confidence: number | null;
  probs: Record<string, number>;
  model_class_name: string;
  model_display_name: string;
  model_confidence: number | null;
  model_probs: Record<string, number>;
  model_version: string;
  rule_version: string;
  decision_reason: string;
  phase_stats: PhaseStats | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export const api = {
  projects: () => request<ProjectInfo[]>("/api/projects"),
  project: (id: string) => request<ProjectInfo>(`/api/projects/${id}`),
  upload: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<ProjectInfo>("/api/projects", { method: "POST", body });
  },
  infer: (id: string) => request<ProjectInfo>(`/api/projects/${id}/infer`, { method: "POST" }),
  cancel: (id: string) => request<ProjectInfo>(`/api/projects/${id}/cancel`, { method: "POST" }),
  reset: (id: string) => request<ProjectInfo>(`/api/projects/${id}/reset`, { method: "POST" }),
  remove: (id: string) => request<{ ok: boolean }>(`/api/projects/${id}`, { method: "DELETE" }),
  threshold: (id: string, threshold: number) =>
    request<ProjectInfo>(`/api/projects/${id}/threshold`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold }),
    }),
  edit: (id: string, mode: "add" | "erase", points: number[][], radius: number) =>
    request<ProjectInfo>(`/api/projects/${id}/edits`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, points, radius }),
    }),
  analyze: (id: string) => request<ClassificationResult>(`/api/projects/${id}/analyze`, { method: "POST" }),
  probabilityAt: (id: string, x: number, y: number) =>
    request<ProbabilityPoint>(`/api/projects/${id}/probability_at?x=${x}&y=${y}`),
  classification: (id: string) => request<ClassificationResult>(`/api/classification/${id}`),
  exportUrl: (id: string) => `${API_BASE}/api/projects/${id}/export`,
  sourceUrl: (id: string) => `${API_BASE}/api/projects/${id}/source`,
  maskUrl: (id: string) => `${API_BASE}/api/projects/${id}/mask?t=${Date.now()}`,
  overlayUrl: (id: string) => `${API_BASE}/api/projects/${id}/overlay?t=${Date.now()}`,
};
