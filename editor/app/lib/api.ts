import type { EditorState, OperationResponse } from "../types";

const API_STORAGE_KEY = "soyrootbio-editor-api";

export function initialApiBase(): string {
  if (typeof window === "undefined") return "";
  const query = new URLSearchParams(window.location.search).get("api");
  if (query) return normalizeApiBase(query);
  return normalizeApiBase(window.localStorage.getItem(API_STORAGE_KEY) ?? "");
}

export function saveApiBase(value: string): string {
  const normalized = normalizeApiBase(value);
  if (typeof window !== "undefined") {
    if (normalized) window.localStorage.setItem(API_STORAGE_KEY, normalized);
    else window.localStorage.removeItem(API_STORAGE_KEY);
  }
  return normalized;
}

export function normalizeApiBase(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

export function apiUrl(apiBase: string, path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${apiBase}${normalizedPath}`;
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = (await response.json().catch(() => ({}))) as {
    error?: string;
  };
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload as T;
}

export async function fetchEditorState(
  apiBase: string,
  signal?: AbortSignal,
): Promise<EditorState> {
  const response = await fetch(apiUrl(apiBase, "/api/state"), {
    signal,
    cache: "no-store",
  });
  return readJson<EditorState>(response);
}

export async function applyEditorOperation(
  apiBase: string,
  type: string,
  args: Record<string, unknown>,
): Promise<OperationResponse> {
  const response = await fetch(apiUrl(apiBase, "/api/operations"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, arguments: args }),
  });
  return readJson<OperationResponse>(response);
}

export async function historyAction(
  apiBase: string,
  action: "undo" | "redo",
): Promise<OperationResponse> {
  const response = await fetch(apiUrl(apiBase, `/api/${action}`), {
    method: "POST",
  });
  return readJson<OperationResponse>(response);
}

export async function exportEdits(apiBase: string): Promise<string> {
  const response = await fetch(apiUrl(apiBase, "/api/export"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const payload = await readJson<{ export_dir: string }>(response);
  return payload.export_dir;
}

export async function fetchLabels(
  apiBase: string,
  path: string,
  expectedVertices: number,
  signal?: AbortSignal,
): Promise<Int32Array> {
  const response = await fetch(apiUrl(apiBase, path), {
    signal,
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Could not load edited labels (${response.status}).`);
  }
  const buffer = await response.arrayBuffer();
  const labels = new Int32Array(buffer);
  if (labels.length !== expectedVertices) {
    throw new Error(
      `Label count ${labels.length.toLocaleString()} does not match the ${expectedVertices.toLocaleString()} mesh vertices.`,
    );
  }
  return labels;
}

export async function fetchPointPatchIndices(
  apiBase: string,
  path: string,
  expectedPoints: number,
  signal?: AbortSignal,
): Promise<Uint32Array> {
  const response = await fetch(apiUrl(apiBase, path), {
    signal,
    cache: "no-store",
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as {
      error?: string;
    };
    throw new Error(
      payload.error || `Could not load the selected point patch (${response.status}).`,
    );
  }
  const buffer = await response.arrayBuffer();
  const indices = new Uint32Array(buffer);
  if (indices.length !== expectedPoints) {
    throw new Error(
      `Patch membership has ${indices.length.toLocaleString()} points; expected ${expectedPoints.toLocaleString()}.`,
    );
  }
  return indices;
}
