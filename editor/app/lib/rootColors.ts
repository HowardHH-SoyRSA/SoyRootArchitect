export type RootColorRgb = readonly [number, number, number];

/**
 * Authoritative SoyRootBio export colors.
 *
 * These values mirror src/soyrootbio/export.py and the color_rgb column written
 * to csv/root_label_map.csv. Keeping one literal browser palette prevents the
 * editor, exported PLY, and exported label map from drifting apart.
 */
export const ROOT_EXPORT_COLORS = {
  unassigned: [140, 140, 140],
  uncertain: [250, 122, 13],
  primary: [13, 59, 224],
  order1: [255, 0, 255],
  order2: [0, 158, 115],
  order3: [140, 51, 209],
  higherOrder: [242, 166, 20],
} as const satisfies Record<string, RootColorRgb>;

export function rootOrderRgb(order: number): RootColorRgb {
  if (order <= 0) return ROOT_EXPORT_COLORS.primary;
  if (order === 1) return ROOT_EXPORT_COLORS.order1;
  if (order === 2) return ROOT_EXPORT_COLORS.order2;
  if (order === 3) return ROOT_EXPORT_COLORS.order3;
  return ROOT_EXPORT_COLORS.higherOrder;
}

export function rootOrderCssColor(order: number): string {
  return rgbToCss(rootOrderRgb(order));
}

export function rootOrderColorNumber(order: number): number {
  return rgbToNumber(rootOrderRgb(order));
}

export function rgbToCss(color: RootColorRgb): string {
  return `#${color
    .map((channel) => channel.toString(16).padStart(2, "0"))
    .join("")}`;
}

export function rgbToNumber([red, green, blue]: RootColorRgb): number {
  return (red << 16) | (green << 8) | blue;
}
