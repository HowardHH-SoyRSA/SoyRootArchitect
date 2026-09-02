/// <reference lib="webworker" />

type ScalarType =
  | "char"
  | "uchar"
  | "short"
  | "ushort"
  | "int"
  | "uint"
  | "float"
  | "double"
  | "int8"
  | "uint8"
  | "int16"
  | "uint16"
  | "int32"
  | "uint32"
  | "float32"
  | "float64";

interface ScalarProperty {
  kind: "scalar";
  name: string;
  type: ScalarType;
}

interface ListProperty {
  kind: "list";
  name: string;
  countType: ScalarType;
  itemType: ScalarType;
}

type Property = ScalarProperty | ListProperty;

interface Element {
  name: string;
  count: number;
  properties: Property[];
}

interface Header {
  format: "ascii" | "binary_little_endian" | "binary_big_endian";
  bodyOffset: number;
  elements: Element[];
}

interface MeshArrays {
  positions: Float32Array;
  colors: Uint8Array;
  labels: Int32Array;
  indices: Uint32Array;
  vertexCount: number;
}

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

workerScope.onmessage = async (
  event: MessageEvent<{
    type: "load";
    url: string;
    renderOrigin: [number, number, number];
  }>,
) => {
  if (event.data.type !== "load") return;
  try {
    postProgress("download", 0, "Downloading full-resolution mesh");
    const data = await download(event.data.url);
    postProgress("parse", 0, "Parsing PLY geometry off the UI thread");
    const parsed = parsePly(data, event.data.renderOrigin);
    postProgress("shading", 0, "Calculating smooth surface normals");
    const normals = computeNormals(
      parsed.positions,
      parsed.indices,
      parsed.vertexCount,
    );
    postProgress("shading", 1, "Surface shading ready");
    workerScope.postMessage(
      {
        type: "result",
        ...parsed,
        normals,
        faceCount: parsed.indices.length / 3,
      },
      {
        transfer: [
          parsed.positions.buffer,
          normals.buffer,
          parsed.indices.buffer,
          parsed.colors.buffer,
          parsed.labels.buffer,
        ],
      },
    );
  } catch (error) {
    workerScope.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};

async function download(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Mesh request failed: ${response.status} ${response.statusText}`);
  }
  const total = Number(response.headers.get("Content-Length") || 0);
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    postProgress("download", 1, "Mesh downloaded");
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    postProgress(
      "download",
      total ? Math.min(received / total, 0.999) : 0,
      total
        ? `Downloading full-resolution mesh · ${formatBytes(received)} / ${formatBytes(total)}`
        : `Downloading full-resolution mesh · ${formatBytes(received)}`,
    );
  }
  const output = new Uint8Array(received);
  let cursor = 0;
  for (const chunk of chunks) {
    output.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  postProgress("download", 1, `Downloaded ${formatBytes(received)}`);
  return output.buffer;
}

function parsePly(
  data: ArrayBuffer,
  renderOrigin: [number, number, number],
): MeshArrays {
  const bytes = new Uint8Array(data);
  const header = parseHeader(bytes);
  return header.format === "ascii"
    ? parseAscii(bytes, header, renderOrigin)
    : parseBinary(data, header, renderOrigin);
}

function parseHeader(bytes: Uint8Array): Header {
  let bodyOffset = -1;
  let lineStart = 0;
  const lines: string[] = [];
  const decoder = new TextDecoder();
  for (let i = 0; i < bytes.length; i += 1) {
    if (bytes[i] !== 10) continue;
    const end = i > lineStart && bytes[i - 1] === 13 ? i - 1 : i;
    const line = decoder.decode(bytes.subarray(lineStart, end)).trim();
    lines.push(line);
    lineStart = i + 1;
    if (line === "end_header") {
      bodyOffset = i + 1;
      break;
    }
  }
  if (bodyOffset < 0 || lines[0] !== "ply") {
    throw new Error("The mesh is not a valid PLY file.");
  }

  let format: Header["format"] | null = null;
  const elements: Element[] = [];
  let current: Element | null = null;
  for (const line of lines) {
    const tokens = line.split(/\s+/);
    if (tokens[0] === "format") {
      const candidate = tokens[1] as Header["format"];
      if (
        candidate !== "ascii" &&
        candidate !== "binary_little_endian" &&
        candidate !== "binary_big_endian"
      ) {
        throw new Error(`Unsupported PLY format: ${tokens[1]}`);
      }
      format = candidate;
    } else if (tokens[0] === "element") {
      current = { name: tokens[1], count: Number(tokens[2]), properties: [] };
      elements.push(current);
    } else if (tokens[0] === "property" && current) {
      if (tokens[1] === "list") {
        current.properties.push({
          kind: "list",
          countType: tokens[2] as ScalarType,
          itemType: tokens[3] as ScalarType,
          name: tokens[4],
        });
      } else {
        current.properties.push({
          kind: "scalar",
          type: tokens[1] as ScalarType,
          name: tokens[2],
        });
      }
    }
  }
  if (!format) throw new Error("PLY header is missing its format declaration.");
  if (!elements.some((element) => element.name === "vertex")) {
    throw new Error("PLY file has no vertex element.");
  }
  return { format, bodyOffset, elements };
}

function parseBinary(
  data: ArrayBuffer,
  header: Header,
  renderOrigin: [number, number, number],
): MeshArrays {
  const littleEndian = header.format === "binary_little_endian";
  const view = new DataView(data);
  const vertexElement = header.elements.find((element) => element.name === "vertex")!;
  const vertexCount = vertexElement.count;
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Uint8Array(vertexCount * 3);
  colors.fill(150);
  const labels = new Int32Array(vertexCount);
  labels.fill(-1);
  let indexCapacity =
    Math.max(
      header.elements.find((element) => element.name === "face")?.count ?? 0,
      1,
    ) * 3;
  let indices = new Uint32Array(indexCapacity);
  let indexCursor = 0;
  let offset = header.bodyOffset;

  const ensureIndexSpace = (needed: number) => {
    if (indexCursor + needed <= indexCapacity) return;
    indexCapacity = Math.max(indexCapacity * 2, indexCursor + needed);
    const expanded = new Uint32Array(indexCapacity);
    expanded.set(indices);
    indices = expanded;
  };

  for (const element of header.elements) {
    const updateEvery = Math.max(1, Math.floor(element.count / 80));
    for (let row = 0; row < element.count; row += 1) {
      if (element.name === "vertex") {
        for (const property of element.properties) {
          if (property.kind === "list") {
            const count = readScalar(view, offset, property.countType, littleEndian);
            offset += scalarSize(property.countType);
            offset += count * scalarSize(property.itemType);
            continue;
          }
          const value = readScalar(view, offset, property.type, littleEndian);
          offset += scalarSize(property.type);
          writeVertexProperty(
            property.name,
            value,
            row,
            positions,
            colors,
            labels,
            renderOrigin,
          );
        }
      } else if (element.name === "face") {
        let faceVertices: number[] | null = null;
        for (const property of element.properties) {
          if (property.kind === "list") {
            const count = readScalar(view, offset, property.countType, littleEndian);
            offset += scalarSize(property.countType);
            const values =
              property.name === "vertex_indices" || property.name === "vertex_index"
                ? new Array<number>(count)
                : null;
            for (let valueIndex = 0; valueIndex < count; valueIndex += 1) {
              const value = readScalar(view, offset, property.itemType, littleEndian);
              offset += scalarSize(property.itemType);
              if (values) values[valueIndex] = value;
            }
            if (values) faceVertices = values;
          } else {
            offset += scalarSize(property.type);
          }
        }
        if (faceVertices && faceVertices.length >= 3) {
          const needed = (faceVertices.length - 2) * 3;
          ensureIndexSpace(needed);
          for (let k = 1; k < faceVertices.length - 1; k += 1) {
            indices[indexCursor++] = faceVertices[0];
            indices[indexCursor++] = faceVertices[k];
            indices[indexCursor++] = faceVertices[k + 1];
          }
        }
      } else {
        offset = skipBinaryElement(view, offset, element.properties, littleEndian);
      }
      if (row % updateEvery === 0) {
        const phaseStart = element.name === "vertex" ? 0 : 0.7;
        const phaseShare = element.name === "vertex" ? 0.7 : 0.3;
        postProgress(
          "parse",
          Math.min(0.99, phaseStart + (row / Math.max(element.count, 1)) * phaseShare),
          element.name === "vertex"
            ? `Reading ${vertexCount.toLocaleString()} vertices`
            : `Building triangle index · ${row.toLocaleString()} / ${element.count.toLocaleString()}`,
        );
      }
    }
  }
  postProgress("parse", 1, "PLY geometry parsed");
  return {
    positions,
    colors,
    labels,
    indices: indices.slice(0, indexCursor),
    vertexCount,
  };
}

function parseAscii(
  bytes: Uint8Array,
  header: Header,
  renderOrigin: [number, number, number],
): MeshArrays {
  const decoder = new TextDecoder();
  const tokens = decoder
    .decode(bytes.subarray(header.bodyOffset))
    .trim()
    .split(/\s+/);
  let tokenCursor = 0;
  const vertexElement = header.elements.find((element) => element.name === "vertex")!;
  const positions = new Float32Array(vertexElement.count * 3);
  const colors = new Uint8Array(vertexElement.count * 3);
  colors.fill(150);
  const labels = new Int32Array(vertexElement.count);
  labels.fill(-1);
  const faceValues: number[] = [];

  for (const element of header.elements) {
    const updateEvery = Math.max(1, Math.floor(element.count / 60));
    for (let row = 0; row < element.count; row += 1) {
      let vertexIndices: number[] | null = null;
      for (const property of element.properties) {
        if (property.kind === "list") {
          const count = Number(tokens[tokenCursor++]);
          const values = tokens
            .slice(tokenCursor, tokenCursor + count)
            .map(Number);
          tokenCursor += count;
          if (
            element.name === "face" &&
            (property.name === "vertex_indices" || property.name === "vertex_index")
          ) {
            vertexIndices = values;
          }
        } else {
          const value = Number(tokens[tokenCursor++]);
          if (element.name === "vertex") {
            writeVertexProperty(
              property.name,
              value,
              row,
              positions,
              colors,
              labels,
              renderOrigin,
            );
          }
        }
      }
      if (vertexIndices && vertexIndices.length >= 3) {
        for (let k = 1; k < vertexIndices.length - 1; k += 1) {
          faceValues.push(vertexIndices[0], vertexIndices[k], vertexIndices[k + 1]);
        }
      }
      if (row % updateEvery === 0) {
        postProgress(
          "parse",
          row / Math.max(element.count, 1),
          `Reading ASCII ${element.name} data`,
        );
      }
    }
  }
  postProgress("parse", 1, "PLY geometry parsed");
  return {
    positions,
    colors,
    labels,
    indices: Uint32Array.from(faceValues),
    vertexCount: vertexElement.count,
  };
}

function writeVertexProperty(
  name: string,
  value: number,
  row: number,
  positions: Float32Array,
  colors: Uint8Array,
  labels: Int32Array,
  renderOrigin: [number, number, number],
) {
  const offset = row * 3;
  if (name === "x") positions[offset] = value - renderOrigin[0];
  else if (name === "y") positions[offset + 1] = value - renderOrigin[1];
  else if (name === "z") positions[offset + 2] = value - renderOrigin[2];
  else if (name === "red" || name === "r") colors[offset] = value;
  else if (name === "green" || name === "g") colors[offset + 1] = value;
  else if (name === "blue" || name === "b") colors[offset + 2] = value;
  else if (name === "root_id") labels[row] = value;
}

function skipBinaryElement(
  view: DataView,
  offset: number,
  properties: Property[],
  littleEndian: boolean,
): number {
  for (const property of properties) {
    if (property.kind === "scalar") {
      offset += scalarSize(property.type);
    } else {
      const count = readScalar(view, offset, property.countType, littleEndian);
      offset += scalarSize(property.countType) + count * scalarSize(property.itemType);
    }
  }
  return offset;
}

function scalarSize(type: ScalarType): number {
  if (type === "char" || type === "uchar" || type === "int8" || type === "uint8") return 1;
  if (type === "short" || type === "ushort" || type === "int16" || type === "uint16") return 2;
  if (
    type === "int" ||
    type === "uint" ||
    type === "float" ||
    type === "int32" ||
    type === "uint32" ||
    type === "float32"
  ) {
    return 4;
  }
  return 8;
}

function readScalar(
  view: DataView,
  offset: number,
  type: ScalarType,
  littleEndian: boolean,
): number {
  if (type === "char" || type === "int8") return view.getInt8(offset);
  if (type === "uchar" || type === "uint8") return view.getUint8(offset);
  if (type === "short" || type === "int16") return view.getInt16(offset, littleEndian);
  if (type === "ushort" || type === "uint16") return view.getUint16(offset, littleEndian);
  if (type === "int" || type === "int32") return view.getInt32(offset, littleEndian);
  if (type === "uint" || type === "uint32") return view.getUint32(offset, littleEndian);
  if (type === "float" || type === "float32") return view.getFloat32(offset, littleEndian);
  return view.getFloat64(offset, littleEndian);
}

function computeNormals(
  positions: Float32Array,
  indices: Uint32Array,
  vertexCount: number,
): Float32Array {
  const normals = new Float32Array(vertexCount * 3);
  const updateEvery = Math.max(3, Math.floor(indices.length / 60 / 3) * 3);
  for (let index = 0; index < indices.length; index += 3) {
    const ia = indices[index] * 3;
    const ib = indices[index + 1] * 3;
    const ic = indices[index + 2] * 3;
    const abx = positions[ib] - positions[ia];
    const aby = positions[ib + 1] - positions[ia + 1];
    const abz = positions[ib + 2] - positions[ia + 2];
    const acx = positions[ic] - positions[ia];
    const acy = positions[ic + 1] - positions[ia + 1];
    const acz = positions[ic + 2] - positions[ia + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    normals[ia] += nx;
    normals[ia + 1] += ny;
    normals[ia + 2] += nz;
    normals[ib] += nx;
    normals[ib + 1] += ny;
    normals[ib + 2] += nz;
    normals[ic] += nx;
    normals[ic + 1] += ny;
    normals[ic + 2] += nz;
    if (index % updateEvery === 0) {
      postProgress(
        "shading",
        index / Math.max(indices.length, 1),
        `Calculating normals · ${(index / 3).toLocaleString()} triangles`,
      );
    }
  }
  for (let index = 0; index < normals.length; index += 3) {
    const length = Math.hypot(normals[index], normals[index + 1], normals[index + 2]) || 1;
    normals[index] /= length;
    normals[index + 1] /= length;
    normals[index + 2] /= length;
  }
  return normals;
}

function postProgress(
  phase: "download" | "parse" | "shading",
  progress: number,
  message: string,
) {
  workerScope.postMessage({ type: "progress", phase, progress, message });
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

export {};
