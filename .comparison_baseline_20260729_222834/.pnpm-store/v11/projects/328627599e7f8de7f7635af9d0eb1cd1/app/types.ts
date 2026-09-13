export type Vec3 = [number, number, number];

export type ToolMode =
  | "select"
  | "create"
  | "split"
  | "merge"
  | "assign"
  | "reconnect"
  | "reparent"
  | "delete"
  | "redraw"
  | "order";

export interface RootRecord {
  root_id: string;
  numeric_label: number;
  parent_id: string | null;
  children_ids: string[];
  root_order: number;
  order_overridden: boolean;
  polyline: Vec3[];
  insertion_point: Vec3;
  tip_point: Vec3;
  length: number | null;
  chord_length: number | null;
  tip_gravity_angle_deg: number | null;
  tip_start_gravity_angle_deg: number | null;
  tip_primary_angle_deg: number | null;
  mean_diameter: number | null;
  surface_area: number | null;
  volume: number | null;
  tortuosity: number | null;
  point_count: number | null;
  confidence: number;
  qc_flags: string[];
  units: {
    length: string;
    area: string;
    volume: string;
    angle: string;
  };
}

export interface PointPatchRecord {
  patch_id: string;
  kind: "uncertain" | "unassigned";
  numeric_label: -2 | -1;
  point_count: number;
  anchor_vertex_index: number;
  centroid: Vec3;
  bounds: {
    minimum: Vec3;
    maximum: Vec3;
  };
  membership_sha256: string;
  revision: number;
  indices_url: string;
}

export interface EditorState {
  schema: string;
  baseline_fingerprint: string;
  source_output_dir: string;
  session_dir: string;
  mesh: {
    vertex_count: number;
    face_count: number;
    root_label_revision: number;
    render_origin: Vec3;
    url: string;
    labels_url: string;
  };
  roots: RootRecord[];
  root_count: number;
  point_patches: PointPatchRecord[];
  point_patch_count: number;
  can_undo: boolean;
  can_redo: boolean;
  operation_count: number;
  supported_operations: string[];
  hardware: {
    logical_cpus: number;
    physical_cpus: number;
    total_memory_bytes: number;
    available_memory_bytes: number;
    gpus: Array<{
      index: number;
      name: string;
      memory_total_bytes: number | null;
      driver_version: string | null;
      backend: string;
      is_discrete: boolean;
    }>;
    discrete_gpu_present: boolean;
    acceleration: {
      rendering: string;
      spatial_picking: string;
      geometry_processing: string;
      full_resolution_policy: string;
    };
  };
}

export interface OperationResponse {
  operation?: {
    operation_id: string;
    sequence: number;
    timestamp: string;
    type: string;
    arguments: Record<string, unknown>;
  };
  state: EditorState;
}

export interface MeshHit {
  rootId: string | null;
  numericLabel: number;
  position: Vec3;
  vertexIndex: number;
}

export interface HoverInfo extends MeshHit {
  clientX: number;
  clientY: number;
}

export interface LoadProgress {
  phase: "idle" | "download" | "parse" | "shading" | "index" | "ready";
  progress: number;
  message: string;
}

export interface ParsedMesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  colors: Uint8Array;
  labels: Int32Array;
  vertexCount: number;
  faceCount: number;
}
