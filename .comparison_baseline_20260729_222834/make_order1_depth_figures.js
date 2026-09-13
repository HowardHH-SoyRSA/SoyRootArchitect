const fs = require("node:fs");
const path = require("node:path");
const sharp = require(
  "C:\\Users\\57340\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules\\.pnpm\\sharp@0.34.5\\node_modules\\sharp"
);

const inputCsv =
  "E:\\Seafile\\Chenlab组会\\20260724\\10 accessions\\output\\BaxiNo2_4-2_20260525\\root_traits.csv";
const outputDir = "E:\\SoyRSA Build\\figure_exports";
const depthBinWidth = 10;
const depthMax = 120;

const width = 1300;
const height = 1040;
const margin = { left: 190, right: 75, top: 125, bottom: 150 };
const plotWidth = width - margin.left - margin.right;
const plotHeight = height - margin.top - margin.bottom;

const colors = {
  blue: "#2878B5",
  grid: "#D7DCE2",
  text: "#222222",
  white: "#FFFFFF",
};

function parseCsv(text) {
  const records = [];
  let record = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      record.push(field);
      field = "";
    } else if (char === "\n") {
      record.push(field.replace(/\r$/, ""));
      records.push(record);
      record = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || record.length) {
    record.push(field.replace(/\r$/, ""));
    records.push(record);
  }

  const headers = records.shift();
  return records
    .filter((row) => row.some((value) => value !== ""))
    .map((row) => Object.fromEntries(headers.map((header, i) => [header, row[i] ?? ""])));
}

function loadOrder1Roots() {
  const text = fs.readFileSync(inputCsv, "utf8").replace(/^\uFEFF/, "");
  const rows = parseCsv(text);
  const primary = rows.find((row) => row.root_id === "primary");
  const baseZ = Number(primary.root_start_z);
  const order1 = rows.filter((row) => row.root_order === "1");
  const data = order1.map((row) => ({
    rootId: row.root_id,
    depth: baseZ - Number(row.root_start_z),
    length: Number(row.length),
    angle: Number(row.tip_gravity_angle_deg),
  }));

  if (data.length !== 57) {
    throw new Error(`Expected 57 order-1 roots, found ${data.length}.`);
  }
  if (data.some((row) => !Number.isFinite(row.depth) || row.depth < 0)) {
    throw new Error("Invalid emergence depth.");
  }
  return data;
}

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function svgText(x, y, text, options = {}) {
  const {
    anchor = "middle",
    size = 23,
    weight = 400,
    rotate = null,
    fill = colors.text,
  } = options;
  const transform = rotate === null ? "" : ` transform="rotate(${rotate} ${x} ${y})"`;
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="Arial, sans-serif" font-size="${size}" font-weight="${weight}" fill="${fill}"${transform}>${esc(text)}</text>`;
}

function baseSvg(title, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="3900" height="3120" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" fill="${colors.white}"/>
  ${svgText(margin.left, 66, title, { anchor: "start", size: 31, weight: 500 })}
  ${body}
</svg>`;
}

function commonDepthElements() {
  const elements = [];
  for (let depth = 0; depth <= depthMax; depth += 10) {
    const y = margin.top + (depth / depthMax) * plotHeight;
    elements.push(
      `<line x1="${margin.left}" y1="${y}" x2="${margin.left + plotWidth}" y2="${y}" stroke="${colors.grid}" stroke-width="1.5"/>`
    );
    elements.push(svgText(margin.left - 20, y + 8, depth, { anchor: "end", size: 20 }));
  }
  elements.push(
    `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="${colors.text}" stroke-width="2"/>`
  );
  elements.push(
    svgText(47, margin.top + plotHeight / 2, "Emergence depth (mesh units)", {
      rotate: -90,
      size: 25,
    })
  );
  return elements;
}

function figureA(data) {
  const counts = Array(depthMax / depthBinWidth).fill(0);
  for (const row of data) {
    const index = Math.min(counts.length - 1, Math.floor(row.depth / depthBinWidth));
    counts[index] += 1;
  }
  const xMax = Math.max(...counts) + 2;
  const elements = commonDepthElements();

  for (let count = 0; count <= xMax; count += 2) {
    const x = margin.left + (count / xMax) * plotWidth;
    elements.push(
      `<line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + plotHeight}" stroke="${colors.grid}" stroke-width="1.5"/>`
    );
    elements.push(svgText(x, margin.top + plotHeight + 40, count, { size: 20 }));
  }
  elements.push(
    `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="${colors.text}" stroke-width="2"/>`
  );

  counts.forEach((count, index) => {
    const y0 = margin.top + ((index * depthBinWidth) / depthMax) * plotHeight;
    const band = (depthBinWidth / depthMax) * plotHeight;
    const barHeight = band * 0.82;
    const y = y0 + (band - barHeight) / 2;
    const barWidth = (count / xMax) * plotWidth;
    elements.push(
      `<rect x="${margin.left}" y="${y}" width="${barWidth}" height="${barHeight}" rx="3" fill="${colors.blue}"/>`
    );
    if (count > 0) {
      elements.push(
        svgText(margin.left + barWidth + 14, y + barHeight / 2 + 8, count, {
          anchor: "start",
          size: 19,
        })
      );
    }
  });
  elements.push(
    svgText(margin.left + plotWidth / 2, height - 55, "Lateral-root density (roots per 10 mesh units)", {
      size: 25,
    })
  );
  elements.push(
    svgText(width - margin.right, 66, `n = ${data.length}`, {
      anchor: "end",
      size: 19,
    })
  );
  return baseSvg("A. Order 1 lateral-root density by depth", elements.join("\n"));
}

function scatterFigure(data, config) {
  const elements = commonDepthElements();
  for (let value = config.xMin; value <= config.xMax; value += config.xStep) {
    const x = margin.left + ((value - config.xMin) / (config.xMax - config.xMin)) * plotWidth;
    elements.push(
      `<line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + plotHeight}" stroke="${colors.grid}" stroke-width="1.5"/>`
    );
    elements.push(svgText(x, margin.top + plotHeight + 40, value, { size: 20 }));
  }
  elements.push(
    `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="${colors.text}" stroke-width="2"/>`
  );

  for (const row of data) {
    const value = row[config.key];
    const x = margin.left + ((value - config.xMin) / (config.xMax - config.xMin)) * plotWidth;
    const y = margin.top + (row.depth / depthMax) * plotHeight;
    elements.push(
      `<circle cx="${x}" cy="${y}" r="8" fill="${colors.blue}" fill-opacity="0.78" stroke="${colors.white}" stroke-width="2"/>`
    );
  }
  elements.push(svgText(margin.left + plotWidth / 2, height - 55, config.xLabel, { size: 25 }));
  elements.push(
    svgText(width - margin.right, 66, `n = ${data.length}`, {
      anchor: "end",
      size: 19,
    })
  );
  return baseSvg(config.title, elements.join("\n"));
}

async function saveJpeg(svg, filename) {
  fs.mkdirSync(outputDir, { recursive: true });
  const output = path.join(outputDir, filename);
  await sharp(Buffer.from(svg))
    .flatten({ background: colors.white })
    .jpeg({ quality: 96, chromaSubsampling: "4:4:4" })
    .withMetadata({ density: 600 })
    .toFile(output);
  return output;
}

async function main() {
  const data = loadOrder1Roots();
  const outputs = [
    await saveJpeg(figureA(data), "A_order1_lateral_root_density_by_depth_600dpi.jpg"),
    await saveJpeg(
      scatterFigure(data, {
        key: "length",
        xMin: 0,
        xMax: 100,
        xStep: 20,
        xLabel: "Lateral-root length (mesh units)",
        title: "B. Order 1 lateral-root length with depth",
      }),
      "B_order1_lateral_root_length_vs_depth_600dpi.jpg"
    ),
    await saveJpeg(
      scatterFigure(data, {
        key: "angle",
        xMin: 0,
        xMax: 180,
        xStep: 30,
        xLabel: "Tip–gravity angle (degrees)",
        title: "C. Order 1 tip–gravity angle with depth",
      }),
      "C_order1_tip_gravity_angle_vs_depth_600dpi.jpg"
    ),
  ];

  for (const output of outputs) {
    const metadata = await sharp(output).metadata();
    if (metadata.format !== "jpeg" || metadata.density !== 600) {
      throw new Error(`Invalid JPEG metadata for ${path.basename(output)}.`);
    }
    if (metadata.width < 3800 || metadata.height < 3000) {
      throw new Error(`Unexpected image size for ${path.basename(output)}.`);
    }
    console.log(
      `${path.basename(output)}\t${metadata.width}x${metadata.height}\t${metadata.density} dpi`
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
