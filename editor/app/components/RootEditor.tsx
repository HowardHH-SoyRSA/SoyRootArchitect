"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  applyEditorOperation,
  exportEdits,
  fetchEditorState,
  historyAction,
  initialApiBase,
  saveApiBase,
} from "../lib/api";
import { useEditorStore } from "../store";
import type { MeshHit, OperationResponse, RootRecord } from "../types";
import {
  ConfirmDialog,
  ConnectionScreen,
  DraftControls,
  HoverTooltip,
  MeshLoading,
  PanelHeader,
  TOOLS,
  Toolbar,
  ToolGuidance,
} from "./EditorChrome";
import { RootColorLegend } from "./RootColorLegend";
import { RootDetails } from "./RootDetails";
import { RootTree } from "./RootTree";
import { RootViewport } from "./RootViewport";

type Toast = {
  id: number;
  tone: "success" | "error" | "info";
  message: string;
};

export function RootEditor() {
  const serverState = useEditorStore((store) => store.serverState);
  const selectedRootId = useEditorStore((store) => store.selectedRootId);
  const selectedPatchId = useEditorStore((store) => store.selectedPatchId);
  const hovered = useEditorStore((store) => store.hovered);
  const tool = useEditorStore((store) => store.tool);
  const draftPoints = useEditorStore((store) => store.draftPoints);
  const brushRadius = useEditorStore((store) => store.brushRadius);
  const loadProgress = useEditorStore((store) => store.loadProgress);
  const meshReady = useEditorStore((store) => store.meshReady);
  const clientGpu = useEditorStore((store) => store.clientGpu);
  const setServerState = useEditorStore((store) => store.setServerState);
  const setSelectedRootId = useEditorStore((store) => store.setSelectedRootId);
  const setSelectedPatchId = useEditorStore(
    (store) => store.setSelectedPatchId,
  );
  const setTool = useEditorStore((store) => store.setTool);
  const addDraftPoint = useEditorStore((store) => store.addDraftPoint);
  const removeDraftPoint = useEditorStore((store) => store.removeDraftPoint);
  const clearDraft = useEditorStore((store) => store.clearDraft);
  const setBrushRadius = useEditorStore((store) => store.setBrushRadius);
  const requestFocus = useEditorStore((store) => store.requestFocus);
  const requestPatchFocus = useEditorStore(
    (store) => store.requestPatchFocus,
  );

  const [apiBase, setApiBase] = useState(() => initialApiBase());
  const [endpointDraft, setEndpointDraft] = useState(() =>
    typeof window === "undefined"
      ? ""
      : initialApiBase() || window.location.origin,
  );
  const [connectionState, setConnectionState] = useState<
    "starting" | "connecting" | "connected" | "error"
  >("connecting");
  const [connectionError, setConnectionError] = useState("");
  const [connectNonce, setConnectNonce] = useState(0);
  const [busy, setBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState("");
  const [deleteCandidate, setDeleteCandidate] = useState<RootRecord | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const toastCounter = useRef(0);
  const scaleInitialized = useRef(false);

  const selectedRoot = useMemo(
    () =>
      serverState?.roots.find((root) => root.root_id === selectedRootId) ?? null,
    [selectedRootId, serverState?.roots],
  );
  const selectedPatch = useMemo(
    () =>
      serverState?.point_patches.find(
        (patch) => patch.patch_id === selectedPatchId,
      ) ?? null,
    [selectedPatchId, serverState?.point_patches],
  );
  const hoveredRoot = useMemo(
    () =>
      serverState?.roots.find((root) => root.root_id === hovered?.rootId) ?? null,
    [hovered?.rootId, serverState?.roots],
  );
  const toolDefinition =
    TOOLS.find((candidate) => candidate.id === tool) ?? TOOLS[0];

  const notify = useCallback(
    (tone: Toast["tone"], message: string) => {
      const id = ++toastCounter.current;
      setToasts((current) => [...current.slice(-2), { id, tone, message }]);
      window.setTimeout(() => {
        setToasts((current) => current.filter((toast) => toast.id !== id));
      }, tone === "error" ? 7000 : 4200);
    },
    [],
  );

  const selectAndFocus = useCallback(
    (rootId: string) => {
      if (
        rootId !== selectedRootId &&
        (tool === "create" || tool === "redraw")
      ) {
        clearDraft();
      }
      setSelectedRootId(rootId);
      requestFocus(rootId);
    },
    [
      clearDraft,
      requestFocus,
      selectedRootId,
      setSelectedRootId,
      tool,
    ],
  );

  const selectPatchAndFocus = useCallback(
    (patchId: string) => {
      setSelectedPatchId(patchId);
      requestPatchFocus(patchId);
    },
    [requestPatchFocus, setSelectedPatchId],
  );

  useEffect(() => {
    const controller = new AbortController();
    fetchEditorState(apiBase, controller.signal)
      .then((state) => {
        setServerState(state);
        setConnectionState("connected");
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setConnectionState("error");
        setConnectionError(
          error instanceof Error
            ? error.message
            : "Could not connect to the editor API.",
        );
      });
    return () => controller.abort();
  }, [apiBase, connectNonce, setServerState]);

  const runOperation = useCallback(
    async (
      operationType: string,
      args: Record<string, unknown>,
      message: string,
    ): Promise<OperationResponse | null> => {
      if (busy) return null;
      setBusy(true);
      setBusyMessage("Applying edit and recomputing root metrics…");
      try {
        const response = await applyEditorOperation(apiBase, operationType, args);
        setServerState(response.state);
        notify("success", message);
        return response;
      } catch (error) {
        notify("error", error instanceof Error ? error.message : String(error));
        return null;
      } finally {
        setBusy(false);
        setBusyMessage("");
      }
    },
    [apiBase, busy, notify, setServerState],
  );

  const runHistory = useCallback(
    async (action: "undo" | "redo") => {
      if (busy) return;
      setBusy(true);
      setBusyMessage(
        `${action === "undo" ? "Undoing" : "Redoing"} edit and recomputing metrics…`,
      );
      try {
        const response = await historyAction(apiBase, action);
        setServerState(response.state);
        clearDraft();
        notify("success", action === "undo" ? "Edit undone." : "Edit restored.");
      } catch (error) {
        notify("error", error instanceof Error ? error.message : String(error));
      } finally {
        setBusy(false);
        setBusyMessage("");
      }
    },
    [
      apiBase,
      busy,
      clearDraft,
      notify,
      setServerState,
    ],
  );

  const handleBrushStroke = useCallback(
    async (hits: MeshHit[]) => {
      if (!selectedRootId) {
        notify("info", "Select a target root before painting points.");
        return;
      }
      if (!hits.length) return;
      await runOperation(
        "assign_points",
        {
          root_id: selectedRootId,
          positions: hits.map((hit) => hit.position),
          radius: brushRadius,
        },
        `Assigned one continuous point region to ${selectedRootId}.`,
      );
    },
    [brushRadius, notify, runOperation, selectedRootId],
  );

  const handleHit = useCallback(
    async (hit: MeshHit) => {
      if (!serverState || busy) return;
      if (tool === "select" || tool === "order") {
        setSelectedRootId(hit.rootId);
        return;
      }
      if (tool === "split") {
        if (!hit.rootId) {
          notify("info", "Choose an assigned root surface for the split.");
          return;
        }
        const response = await runOperation(
          "split_root",
          { root_id: hit.rootId, position: hit.position },
          `Split ${hit.rootId}.`,
        );
        const newRootId = response?.operation?.arguments.new_root_id;
        if (typeof newRootId === "string") setSelectedRootId(newRootId);
        if (response) setTool("select");
        return;
      }
      if (tool === "delete") {
        const candidate = serverState.roots.find(
          (root) => root.root_id === hit.rootId,
        );
        if (!candidate) {
          notify("info", "Choose an assigned root to delete.");
        } else if (candidate.root_id === "primary") {
          notify("error", "The primary root cannot be deleted.");
        } else {
          setDeleteCandidate(candidate);
        }
        return;
      }
      if (tool === "redraw") {
        if (!selectedRootId) {
          notify("info", "Select the root to redraw in the tree first.");
        } else {
          addDraftPoint(hit.position);
        }
        return;
      }
      if (tool === "create") {
        if (!selectedRootId) {
          notify("info", "Select the parent root in the hierarchy first.");
        } else if (hit.numericLabel !== -1) {
          notify(
            "info",
            "New roots can be drawn only through grey unassigned surface points.",
          );
        } else {
          addDraftPoint(hit.position);
        }
        return;
      }
      if (!selectedRootId) {
        notify(
          "info",
          `Select a source root before using ${toolDefinition.label}.`,
        );
        return;
      }
      if (tool === "assign") {
        await handleBrushStroke([hit]);
        return;
      }
      if (!hit.rootId) {
        notify("info", "Choose another assigned root as the target.");
        return;
      }
      if (hit.rootId === selectedRootId) {
        notify("info", "Choose a different target root.");
        return;
      }
      if (tool === "merge") {
        if (selectedRootId === "primary" || hit.rootId === "primary") {
          notify("error", "The primary root cannot be merged.");
          return;
        }
        const response = await runOperation(
          "merge_roots",
          { root_id: selectedRootId, other_root_id: hit.rootId },
          `Merged ${hit.rootId} into ${selectedRootId}.`,
        );
        if (response) setTool("select");
      } else if (tool === "reconnect") {
        const response = await runOperation(
          "reconnect_root",
          {
            root_id: selectedRootId,
            target_root_id: hit.rootId,
            position: hit.position,
          },
          `Reconnected ${selectedRootId} to ${hit.rootId}.`,
        );
        if (response) setTool("select");
      } else if (tool === "reparent") {
        const response = await runOperation(
          "reparent_root",
          {
            root_id: selectedRootId,
            new_parent_id: hit.rootId,
            position: hit.position,
          },
          `Reparented ${selectedRootId} to ${hit.rootId}.`,
        );
        if (response) setTool("select");
      }
    },
    [
      addDraftPoint,
      busy,
      handleBrushStroke,
      notify,
      runOperation,
      selectedRootId,
      serverState,
      setSelectedRootId,
      setTool,
      tool,
      toolDefinition.label,
    ],
  );

  const applyRedraw = useCallback(async () => {
    if (!selectedRootId || draftPoints.length < 2) {
      notify("info", "A redrawn path needs at least two points.");
      return;
    }
    const response = await runOperation(
      "redraw_root",
      { root_id: selectedRootId, points: draftPoints },
      `Redrew ${selectedRootId} with ${draftPoints.length} path points.`,
    );
    if (response) {
      clearDraft();
      setTool("select");
    }
  }, [
    clearDraft,
    draftPoints,
    notify,
    runOperation,
    selectedRootId,
    setTool,
  ]);

  const applyCreate = useCallback(async () => {
    if (!selectedRootId || draftPoints.length < 2) {
      notify(
        "info",
        "Select a parent and draw through at least two unassigned points.",
      );
      return;
    }
    const response = await runOperation(
      "create_root",
      {
        parent_id: selectedRootId,
        points: draftPoints,
        claim_radius: brushRadius,
      },
      `Created a root from ${draftPoints.length} path points.`,
    );
    const newRootId = response?.operation?.arguments.new_root_id;
    if (response && typeof newRootId === "string") {
      clearDraft();
      setTool("select");
      setSelectedRootId(newRootId);
    }
  }, [
    brushRadius,
    clearDraft,
    draftPoints,
    notify,
    runOperation,
    selectedRootId,
    setSelectedRootId,
    setTool,
  ]);

  const confirmDelete = useCallback(async () => {
    if (!deleteCandidate) return;
    const rootId = deleteCandidate.root_id;
    setDeleteCandidate(null);
    const response = await runOperation(
      "delete_root",
      { root_id: rootId },
      `Deleted ${rootId}. The automatic source remains untouched.`,
    );
    if (response) {
      setSelectedRootId(null);
      setTool("select");
    }
  }, [deleteCandidate, runOperation, setSelectedRootId, setTool]);

  const applyOrder = useCallback(
    async (order: number) => {
      if (!selectedRoot) return;
      const response = await runOperation(
        "correct_root_order",
        { root_id: selectedRoot.root_id, root_order: order },
        `Set ${selectedRoot.root_id} to order ${order}.`,
      );
      if (response) setTool("select");
    },
    [runOperation, selectedRoot, setTool],
  );

  const handleExport = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setBusyMessage("Materialising edited files beside the operation log…");
    try {
      const path = await exportEdits(apiBase);
      notify("success", `Edited files exported to ${path}`);
    } catch (error) {
      notify("error", error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
      setBusyMessage("");
    }
  }, [apiBase, busy, notify]);

  const handleScale = useCallback(
    (scale: number) => {
      if (scaleInitialized.current) return;
      scaleInitialized.current = true;
      setBrushRadius(Math.max(scale * 0.012, 0.001));
    },
    [setBrushRadius],
  );
  const handleViewportError = useCallback(
    (message: string) => notify("error", message),
    [notify],
  );

  const handleEndpointSubmit = (event: FormEvent) => {
    event.preventDefault();
    let candidate = endpointDraft.trim();
    if (candidate === window.location.origin) candidate = "";
    const normalized = saveApiBase(candidate);
    setConnectionState("connecting");
    setConnectionError("");
    setApiBase(normalized);
    setConnectNonce((value) => value + 1);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const editing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        void runHistory(event.shiftKey ? "redo" : "undo");
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        void runHistory("redo");
        return;
      }
      if (event.key === "Escape") {
        setDeleteCandidate(null);
        clearDraft();
        setTool("select");
        return;
      }
      if (editing || event.ctrlKey || event.metaKey || event.altKey) return;
      const selectedTool = TOOLS.find(
        (candidate) => candidate.shortcut === event.key,
      );
      if (selectedTool) setTool(selectedTool.id);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clearDraft, runHistory, setTool]);

  if (connectionState !== "connected" || !serverState) {
    return (
      <ConnectionScreen
        state={connectionState}
        error={connectionError}
        endpoint={endpointDraft}
        onEndpointChange={setEndpointDraft}
        onSubmit={handleEndpointSubmit}
        onRetry={() => {
          saveApiBase("");
          setConnectionState("connecting");
          setConnectionError("");
          setApiBase("");
          setEndpointDraft(window.location.origin);
          setConnectNonce((value) => value + 1);
        }}
      />
    );
  }

  const sampleName = leafName(serverState.source_output_dir);
  const hostGpu = serverState.hardware.gpus[0];
  const gpuName = hostGpu?.name || clientGpu || "WebGL graphics";
  const hardwareAudit = [
    `Host GPU: ${hostGpu?.name ?? "no discrete GPU reported"}${
      hostGpu?.memory_total_bytes
        ? ` (${formatBytes(hostGpu.memory_total_bytes)})`
        : ""
    }`,
    `Browser renderer: ${clientGpu ?? "detecting…"}`,
    `CPU: ${serverState.hardware.logical_cpus} logical / ${serverState.hardware.physical_cpus} physical`,
    `RAM available: ${formatBytes(serverState.hardware.available_memory_bytes)} / ${formatBytes(serverState.hardware.total_memory_bytes)}`,
    `Policy: ${serverState.hardware.acceleration.full_resolution_policy}`,
  ].join("\n");

  return (
    <main className="editor-app">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span /><span /><span />
          </div>
          <div><strong>SoyRoot Studio</strong><span>3D graph editor</span></div>
        </div>
        <div className="dataset-heading" title={serverState.source_output_dir}>
          <span>ACTIVE DATASET</span><strong>{sampleName}</strong>
        </div>
        <div className="header-status">
          <div className="status-chip immutable-chip" title="Automatic output files are read-only">
            <span className="lock-glyph">▣</span>Source preserved
          </div>
          <div className="status-chip gpu-chip" title={hardwareAudit}>
            <span className="live-dot" />{compactGpuName(gpuName)}
          </div>
          <div className="operation-count" title="Active edits in this session">
            <strong>{serverState.operation_count}</strong><span>edits</span>
          </div>
        </div>
      </header>

      <Toolbar
        activeTool={tool}
        state={serverState}
        busy={busy}
        onTool={setTool}
        onUndo={() => void runHistory("undo")}
        onRedo={() => void runHistory("redo")}
        onExport={() => void handleExport()}
      />

      <section className={`workspace ${leftCollapsed ? "left-collapsed" : ""} ${rightCollapsed ? "right-collapsed" : ""}`}>
        <aside className="panel tree-panel" aria-label="Root hierarchy">
          <PanelHeader eyebrow="STRUCTURE" title={`${serverState.root_count} roots · ${serverState.point_patch_count} patches`} side="left" collapsed={leftCollapsed} onCollapse={() => setLeftCollapsed((value) => !value)} />
          <RootColorLegend collapsed={leftCollapsed} />
          {!leftCollapsed ? (
            <RootTree
              roots={serverState.roots}
              pointPatches={serverState.point_patches}
              selectedRootId={selectedRootId}
              selectedPatchId={selectedPatchId}
              onSelect={selectAndFocus}
              onSelectPatch={selectPatchAndFocus}
            />
          ) : null}
        </aside>

        <section className="viewport-region" aria-label="3D viewport">
          <RootViewport
            apiBase={apiBase}
            state={serverState}
            interactionLocked={busy || !meshReady}
            onHit={handleHit}
            onStroke={handleBrushStroke}
            onError={handleViewportError}
            onScaleChange={handleScale}
          />
          <ToolGuidance tool={toolDefinition} />
          {!meshReady ? <MeshLoading progress={loadProgress} /> : null}
          {tool === "redraw" || tool === "create" ? (
            <DraftControls
              mode={tool}
              count={draftPoints.length}
              disabled={busy}
              selectedRoot={selectedRoot}
              onUndoPoint={removeDraftPoint}
              onClear={clearDraft}
              onApply={() =>
                void (tool === "create" ? applyCreate() : applyRedraw())
              }
            />
          ) : null}
        </section>

        <aside className="panel details-panel" aria-label="Root information">
          <PanelHeader eyebrow="INSPECTOR" title={selectedPatch ? selectedPatch.patch_id : selectedRoot ? selectedRoot.root_id : "No selection"} side="right" collapsed={rightCollapsed} onCollapse={() => setRightCollapsed((value) => !value)} />
          {!rightCollapsed ? (
            <RootDetails
              root={selectedRoot}
              patch={selectedPatch}
              roots={serverState.roots}
              activeTool={tool}
              brushRadius={brushRadius}
              busy={busy}
              onBrushRadius={setBrushRadius}
              onApplyOrder={(order) => void applyOrder(order)}
              onSelect={selectAndFocus}
            />
          ) : null}
        </aside>
      </section>

      <footer className="app-footer">
        <span><i className="footer-dot" />Full-resolution session</span>
        <span>{serverState.mesh.face_count.toLocaleString()} faces</span>
        <span>History is append-only · <kbd>Ctrl Z</kbd> undo · <kbd>Ctrl Shift Z</kbd> redo</span>
        <span className="fingerprint" title={serverState.baseline_fingerprint}>{serverState.baseline_fingerprint.slice(0, 19)}…</span>
      </footer>

      {hovered ? (
        <HoverTooltip hovered={hovered} root={hoveredRoot} activeTool={tool} />
      ) : null}

      {busy ? (
        <div className="busy-curtain" role="status" aria-live="assertive">
          <div className="spinner" /><strong>{busyMessage}</strong>
          <span>The automatic result is not being overwritten.</span>
        </div>
      ) : null}

      {deleteCandidate ? (
        <ConfirmDialog root={deleteCandidate} onCancel={() => setDeleteCandidate(null)} onConfirm={() => void confirmDelete()} />
      ) : null}

      <div className="toast-stack" aria-live="polite">
        {toasts.map((toast) => (
          <div className={`toast toast-${toast.tone}`} key={toast.id}>
            <span>{toast.tone === "success" ? "✓" : toast.tone === "error" ? "!" : "i"}</span>
            <p>{toast.message}</p>
          </div>
        ))}
      </div>
    </main>
  );
}

function leafName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path;
}

function compactGpuName(name: string): string {
  return name
    .replace(/NVIDIA/gi, "")
    .replace(/GeForce/gi, "")
    .replace(/ANGLE\s*\(/gi, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 24);
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "unknown";
  const gib = bytes / 1024 ** 3;
  return gib >= 1 ? `${gib.toFixed(1)} GiB` : `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
}
