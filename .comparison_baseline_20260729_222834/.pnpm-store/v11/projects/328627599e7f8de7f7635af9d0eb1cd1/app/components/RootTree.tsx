"use client";

import { useMemo, useState, type CSSProperties } from "react";
import {
  ROOT_EXPORT_COLORS,
  rgbToCss,
  rootOrderCssColor,
} from "../lib/rootColors";
import type { PointPatchRecord, RootRecord } from "../types";
import { formatMetric } from "./EditorChrome";

export function RootTree({
  roots,
  pointPatches,
  selectedRootId,
  selectedPatchId,
  onSelect,
  onSelectPatch,
}: {
  roots: RootRecord[];
  pointPatches: PointPatchRecord[];
  selectedRootId: string | null;
  selectedPatchId: string | null;
  onSelect: (rootId: string) => void;
  onSelectPatch: (patchId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const normalizedQuery = query.trim().toLowerCase();
  const rows = useMemo(
    () =>
      normalizedQuery
        ? roots
            .filter(
              (root) =>
                root.root_id.toLowerCase().includes(normalizedQuery) ||
                String(root.root_order) === normalizedQuery,
            )
            .map((root) => ({ root, depth: 0 }))
        : flattenRoots(roots, collapsed),
    [collapsed, normalizedQuery, roots],
  );
  const visiblePatches = useMemo(
    () =>
      pointPatches.filter(
        (patch) =>
          !normalizedQuery ||
          patch.patch_id.toLowerCase().includes(normalizedQuery) ||
          patch.kind.includes(normalizedQuery),
      ),
    [normalizedQuery, pointPatches],
  );
  const uncertainPatches = visiblePatches.filter(
    (patch) => patch.kind === "uncertain",
  );
  const unassignedPatches = visiblePatches.filter(
    (patch) => patch.kind === "unassigned",
  );

  const toggle = (rootId: string) => {
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(rootId)) next.delete(rootId);
      else next.add(rootId);
      return next;
    });
  };

  return (
    <div className="tree-body">
      <label className="search-field">
        <span aria-hidden="true">⌕</span>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search roots or patches"
          aria-label="Search roots or point patches"
        />
        {query ? (
          <button
            type="button"
            onClick={() => setQuery("")}
            aria-label="Clear search"
          >
            ×
          </button>
        ) : null}
      </label>

      <div className="tree-list">
        <div role="tree" aria-label="Root hierarchy">
          {rows.map(({ root, depth }) => {
            const hasChildren = root.children_ids.length > 0;
            const isCollapsed = collapsed.has(root.root_id);
            const selected =
              selectedRootId === root.root_id && !selectedPatchId;
            return (
              <div
                className={`tree-row ${selected ? "selected" : ""}`}
                key={root.root_id}
                role="treeitem"
                aria-selected={selected}
                aria-expanded={hasChildren ? !isCollapsed : undefined}
                style={{ "--tree-depth": depth } as CSSProperties}
              >
                <button
                  type="button"
                  className="tree-chevron"
                  onClick={() => hasChildren && toggle(root.root_id)}
                  aria-label={
                    hasChildren
                      ? `${isCollapsed ? "Expand" : "Collapse"} ${root.root_id}`
                      : undefined
                  }
                  tabIndex={hasChildren ? 0 : -1}
                >
                  {hasChildren ? (isCollapsed ? "›" : "⌄") : "·"}
                </button>
                <button
                  type="button"
                  className="tree-root-button"
                  onClick={() => onSelect(root.root_id)}
                >
                  <i
                    className="root-order-dot"
                    style={{ background: rootOrderCssColor(root.root_order) }}
                  />
                  <span>
                    <b>{root.root_id}</b>
                    <small>
                      order {root.root_order} · {formatMetric(root.length, 2)}
                    </small>
                  </span>
                  {root.qc_flags.length ? (
                    <em title={root.qc_flags.join(", ")}>!</em>
                  ) : null}
                </button>
              </div>
            );
          })}
        </div>

        <PatchSection
          title="Uncertain patches"
          patches={uncertainPatches}
          color={rgbToCss(ROOT_EXPORT_COLORS.uncertain)}
          selectedPatchId={selectedPatchId}
          onSelect={onSelectPatch}
        />
        <PatchSection
          title="Unassigned patches"
          patches={unassignedPatches}
          color={rgbToCss(ROOT_EXPORT_COLORS.unassigned)}
          selectedPatchId={selectedPatchId}
          onSelect={onSelectPatch}
        />

        {!rows.length && !visiblePatches.length ? (
          <p className="tree-empty">
            No roots or patches match “{query}”.
          </p>
        ) : null}
      </div>

      <p className="tree-hint">
        Choose a root or a connected uncertain/unassigned patch to highlight
        and frame it.
      </p>
    </div>
  );
}

function PatchSection({
  title,
  patches,
  color,
  selectedPatchId,
  onSelect,
}: {
  title: string;
  patches: PointPatchRecord[];
  color: string;
  selectedPatchId: string | null;
  onSelect: (patchId: string) => void;
}) {
  return (
    <section className="patch-section" aria-label={title}>
      <div className="patch-section-heading">
        <span>{title}</span>
        <b>{patches.length}</b>
      </div>
      <div role="listbox" aria-label={title}>
        {patches.map((patch) => {
          const selected = selectedPatchId === patch.patch_id;
          const shortId = patch.patch_id
            .replace("uncertain-", "Q-")
            .replace("unassigned-", "U-");
          return (
            <button
              type="button"
              className={`patch-row ${selected ? "selected" : ""}`}
              key={patch.patch_id}
              role="option"
              aria-selected={selected}
              onClick={() => onSelect(patch.patch_id)}
            >
              <i style={{ background: color }} />
              <span>
                <b>{shortId}</b>
                <small>
                  {patch.point_count.toLocaleString()}{" "}
                  {patch.point_count === 1 ? "point" : "points"}
                </small>
              </span>
              <em>→</em>
            </button>
          );
        })}
        {!patches.length ? (
          <p className="patch-empty">No {title.toLowerCase()}.</p>
        ) : null}
      </div>
    </section>
  );
}

function flattenRoots(
  roots: RootRecord[],
  collapsed: Set<string>,
): Array<{ root: RootRecord; depth: number }> {
  const rootIds = new Set(roots.map((root) => root.root_id));
  const byParent = new Map<string | null, RootRecord[]>();
  for (const root of roots) {
    const parent =
      root.parent_id && rootIds.has(root.parent_id) ? root.parent_id : null;
    const siblings = byParent.get(parent) ?? [];
    siblings.push(root);
    byParent.set(parent, siblings);
  }
  for (const siblings of byParent.values()) {
    siblings.sort(
      (a, b) =>
        a.root_order - b.root_order || a.root_id.localeCompare(b.root_id),
    );
  }
  const output: Array<{ root: RootRecord; depth: number }> = [];
  const visited = new Set<string>();
  const visit = (root: RootRecord, depth: number) => {
    if (visited.has(root.root_id)) return;
    visited.add(root.root_id);
    output.push({ root, depth });
    if (!collapsed.has(root.root_id)) {
      for (const child of byParent.get(root.root_id) ?? []) {
        visit(child, depth + 1);
      }
    }
  };
  const primary = roots.find((root) => root.root_id === "primary");
  if (primary) visit(primary, 0);
  for (const root of byParent.get(null) ?? []) visit(root, 0);
  for (const root of roots) visit(root, 0);
  return output;
}
