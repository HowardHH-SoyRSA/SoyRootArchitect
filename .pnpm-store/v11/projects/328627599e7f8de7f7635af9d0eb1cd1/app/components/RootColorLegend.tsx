import { ROOT_EXPORT_COLORS, rgbToCss } from "../lib/rootColors";

const LEGEND_ITEMS = [
  { label: "Primary", order: "O0", color: ROOT_EXPORT_COLORS.primary },
  { label: "First order", order: "O1", color: ROOT_EXPORT_COLORS.order1 },
  { label: "Second order", order: "O2", color: ROOT_EXPORT_COLORS.order2 },
  { label: "Third order", order: "O3", color: ROOT_EXPORT_COLORS.order3 },
  { label: "Higher order", order: "O4+", color: ROOT_EXPORT_COLORS.higherOrder },
  { label: "Uncertain", order: "QC", color: ROOT_EXPORT_COLORS.uncertain },
  { label: "Unassigned", order: "—", color: ROOT_EXPORT_COLORS.unassigned },
] as const;

export function RootColorLegend({ collapsed }: { collapsed: boolean }) {
  return (
    <aside
      className={`root-color-legend ${collapsed ? "is-collapsed" : ""}`}
      aria-label="SoyRootBio exported color legend"
    >
      <header>
        <span>EXPORT COLORS</span>
        <small>PLY · CSV</small>
      </header>
      <ul>
        {LEGEND_ITEMS.map((item) => {
          const color = rgbToCss(item.color);
          const description = `${item.label} · ${item.order} · ${color.toUpperCase()}`;
          return (
            <li key={item.label} aria-label={description} title={description}>
              <i aria-hidden="true" style={{ background: color }} />
              <span>{item.label}<b>{item.order}</b></span>
              <code>{color.toUpperCase()}</code>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
