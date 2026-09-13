"use client";

import { useState } from "react";
import {
  ROOT_EXPORT_COLORS,
  rgbToCss,
  rootOrderCssColor,
} from "../lib/rootColors";
import type { PointPatchRecord, RootRecord, ToolMode } from "../types";
import { formatMetric } from "./EditorChrome";

export function RootDetails({
  root,
  patch,
  roots,
  activeTool,
  brushRadius,
  busy,
  onBrushRadius,
  onApplyOrder,
  onSelect,
}: {
  root: RootRecord | null;
  patch: PointPatchRecord | null;
  roots: RootRecord[];
  activeTool: ToolMode;
  brushRadius: number;
  busy: boolean;
  onBrushRadius: (radius: number) => void;
  onApplyOrder: (order: number) => void;
  onSelect: (rootId: string) => void;
}) {
  if (patch) {
    return <PointPatchDetails patch={patch} editTarget={root} />;
  }
  if (!root) {
    return (
      <div className="empty-inspector">
        <div className="empty-root-icon"><span /><span /><span /></div>
        <h3>Inspect a root</h3>
        <p>Click the 3D surface or choose a root from the hierarchy to reveal its measurements and direct relationships.</p>
      </div>
    );
  }

  const parent = roots.find((candidate) => candidate.root_id === root.parent_id);
  return (
    <div className="details-scroll">
      <section className="identity-block">
        <div className="root-title-line">
          <div>
            <span
              className="order-pill"
              style={{
                borderColor: rootOrderCssColor(root.root_order),
                color: rootOrderCssColor(root.root_order),
                background: `${rootOrderCssColor(root.root_order)}18`,
              }}
            >
              ORDER {root.root_order}
            </span>
            {root.order_overridden ? <span className="manual-pill">MANUAL</span> : null}
          </div>
          <button type="button" className="copy-button" onClick={() => void navigator.clipboard?.writeText(root.root_id)} title="Copy root ID">Copy ID</button>
        </div>
        <h2>{root.root_id}</h2>
        <p>{root.polyline.length.toLocaleString()} centerline nodes · {root.point_count?.toLocaleString() ?? "—"} assigned mesh points</p>
      </section>

      {activeTool === "assign" || activeTool === "create" || activeTool === "reparent" || activeTool === "order" ? (
        <section className="tool-options">
          <span className="section-label">TOOL OPTIONS</span>
          {activeTool === "assign" || activeTool === "create" ? (
            <label>
              <span>
                {activeTool === "create" ? "Path claim radius" : "3D brush radius"}
                {" "}<b>{formatMetric(brushRadius, 3)}</b>
              </span>
              <input
                type="range"
                min={Math.max(brushRadius / 12, 0.0001)}
                max={Math.max(brushRadius * 8, 0.001)}
                step={Math.max(brushRadius / 50, 0.00001)}
                value={brushRadius}
                onChange={(event) => onBrushRadius(Number(event.target.value))}
              />
              <input
                className="number-input"
                type="number"
                min="0.000001"
                step="any"
                value={brushRadius}
                onChange={(event) => onBrushRadius(Math.max(Number(event.target.value), 0.000001))}
              />
            </label>
          ) : null}
          {activeTool === "create" ? (
            <p className="connection-rule">
              This selected root will be the parent. Click at least two grey
              unassigned surface points from base to tip; nearby unassigned
              points will be claimed along the resulting path.
            </p>
          ) : null}
          {activeTool === "assign" ? (
            <p className="connection-rule">
              Hold Shift and left-drag across the surface to paint one
              continuous point region. The full stroke is stored as one
              undoable operation; left-drag rotates normally.
            </p>
          ) : null}
          {activeTool === "reparent" ? (
            <p className="connection-rule">
              Click the intended attachment on the new parent. The root base is
              moved to that exact graph connection so topology and geometry stay
              consistent.
            </p>
          ) : null}
          {activeTool === "order" ? (
            <OrderEditor
              key={`${root.root_id}:${root.root_order}`}
              root={root}
              busy={busy}
              onApplyOrder={onApplyOrder}
            />
          ) : null}
        </section>
      ) : null}

      <section className="metric-section">
        <span className="section-label">MORPHOMETRY</span>
        <div className="metric-grid">
          <Metric label="Path length" value={root.length} unit={root.units.length} />
          <Metric label="Chord" value={root.chord_length} unit={root.units.length} />
          <Metric label="Mean diameter" value={root.mean_diameter} unit={root.units.length} />
          <Metric label="Tortuosity" value={root.tortuosity} />
          <Metric label="Surface area" value={root.surface_area} unit={root.units.area} wide />
          <Metric label="Volume" value={root.volume} unit={root.units.volume} wide />
        </div>
      </section>

      <section className="metric-section">
        <span className="section-label">ORIENTATION</span>
        <div className="angle-list">
          <AngleMetric label="Tip ↔ gravity" value={root.tip_gravity_angle_deg} />
          <AngleMetric label="Tip-start ↔ gravity" value={root.tip_start_gravity_angle_deg} />
          <AngleMetric label="Lateral-start → primary" value={root.tip_primary_angle_deg} />
        </div>
      </section>

      <section className="metric-section">
        <span className="section-label">GRAPH RELATIONSHIPS</span>
        <div className="relation-row">
          <span>Parent</span>
          {parent ? (
            <button type="button" onClick={() => onSelect(parent.root_id)}>
              <i style={{ background: rootOrderCssColor(parent.root_order) }} />{parent.root_id}<b>↗</b>
            </button>
          ) : <em>None · primary axis</em>}
        </div>
        <details className="children-disclosure">
          <summary><span>Direct children</span><b>{root.children_ids.length}</b></summary>
          <div>
            {root.children_ids.length ? root.children_ids.map((childId) => {
              const child = roots.find((candidate) => candidate.root_id === childId);
              return (
                <button type="button" key={childId} onClick={() => onSelect(childId)}>
                  <i style={{ background: child ? rootOrderCssColor(child.root_order) : rgbToCss(ROOT_EXPORT_COLORS.unassigned) }} />{childId}<span>↗</span>
                </button>
              );
            }) : <p>No direct children.</p>}
          </div>
        </details>
      </section>

      <section className="metric-section">
        <span className="section-label">LANDMARKS</span>
        <Coordinate label="Insertion" color={rgbToCss(ROOT_EXPORT_COLORS.higherOrder)} value={root.insertion_point} />
        <Coordinate label="Tip" color="#ff5c64" value={root.tip_point} />
      </section>

      {root.qc_flags.length ? (
        <section className="metric-section">
          <span className="section-label">QUALITY NOTES</span>
          <div className="flag-list">{root.qc_flags.map((flag) => <span key={flag}>{flag.replaceAll("_", " ")}</span>)}</div>
        </section>
      ) : null}
    </div>
  );
}

function PointPatchDetails({
  patch,
  editTarget,
}: {
  patch: PointPatchRecord;
  editTarget: RootRecord | null;
}) {
  const color = rgbToCss(
    patch.kind === "uncertain"
      ? ROOT_EXPORT_COLORS.uncertain
      : ROOT_EXPORT_COLORS.unassigned,
  );
  const shortId = patch.patch_id
    .replace("uncertain-", "Q-")
    .replace("unassigned-", "U-");
  return (
    <div className="details-scroll">
      <section className="identity-block">
        <div className="root-title-line">
          <span
            className="order-pill"
            style={{
              borderColor: color,
              color,
              background: `${color}18`,
            }}
          >
            {patch.kind.toUpperCase()} PATCH
          </span>
        </div>
        <h2>{shortId}</h2>
        <p>
          {patch.point_count.toLocaleString()} edge-connected mesh{" "}
          {patch.point_count === 1 ? "point" : "points"}
        </p>
        {editTarget ? (
          <div className="patch-edit-target">
            <span>Current edit target</span>
            <b>{editTarget.root_id}</b>
          </div>
        ) : null}
      </section>

      <section className="metric-section">
        <span className="section-label">PATCH LOCATION</span>
        <Coordinate label="Centroid" color={color} value={patch.centroid} />
        <Coordinate label="Bounds min" color={color} value={patch.bounds.minimum} />
        <Coordinate label="Bounds max" color={color} value={patch.bounds.maximum} />
      </section>

      <section className="metric-section">
        <span className="section-label">CONNECTIVITY</span>
        <div className="relation-row">
          <span>Definition</span>
          <em>Shared triangle edges</em>
        </div>
        <div className="relation-row">
          <span>Anchor vertex</span>
          <em>{patch.anchor_vertex_index.toLocaleString()}</em>
        </div>
      </section>
    </div>
  );
}

function OrderEditor({
  root,
  busy,
  onApplyOrder,
}: {
  root: RootRecord;
  busy: boolean;
  onApplyOrder: (order: number) => void;
}) {
  const [orderValue, setOrderValue] = useState(root.root_order);
  return (
    <div className="order-editor">
      <label>
        Correct root order
        <input
          className="number-input"
          type="number"
          min={root.root_id === "primary" ? 0 : 1}
          max={root.root_id === "primary" ? 0 : 253}
          value={orderValue}
          onChange={(event) => setOrderValue(Number(event.target.value))}
        />
      </label>
      <button
        type="button"
        className="small-apply"
        disabled={busy || orderValue === root.root_order}
        onClick={() => onApplyOrder(orderValue)}
      >
        Apply order
      </button>
    </div>
  );
}

function Metric({ label, value, unit, wide }: { label: string; value: number | null; unit?: string; wide?: boolean }) {
  return <div className={`metric-card ${wide ? "wide" : ""}`}><span>{label}</span><strong>{formatMetric(value, 3)}</strong>{unit ? <small>{unit}</small> : null}</div>;
}

function AngleMetric({ label, value }: { label: string; value: number | null }) {
  const finite = typeof value === "number" && Number.isFinite(value);
  return (
    <div>
      <span>{label}</span>
      <div className="angle-track"><i style={{ width: finite ? `${Math.min(Math.abs(value) / 180, 1) * 100}%` : "0%" }} /></div>
      <strong>{finite ? `${value.toFixed(2)}°` : "—"}</strong>
    </div>
  );
}

function Coordinate({ label, color, value }: { label: string; color: string; value: [number, number, number] }) {
  return <div className="coordinate-row"><span><i style={{ background: color }} /> {label}</span><code>{value.map((coordinate) => formatMetric(coordinate, 2)).join(", ")}</code></div>;
}
