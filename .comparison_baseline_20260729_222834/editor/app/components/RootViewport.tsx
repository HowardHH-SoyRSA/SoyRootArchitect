"use client";

import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { acceleratedRaycast, SAH, type MeshBVH } from "three-mesh-bvh";
import { GenerateMeshBVHWorker } from "three-mesh-bvh/worker";
import { apiUrl, fetchLabels, fetchPointPatchIndices } from "../lib/api";
import {
  ROOT_EXPORT_COLORS,
  rgbToNumber,
  rootOrderColorNumber,
  rootOrderRgb,
} from "../lib/rootColors";
import { useEditorStore } from "../store";
import type {
  EditorState,
  MeshHit,
  ParsedMesh,
  PointPatchRecord,
  RootRecord,
  ToolMode,
  Vec3,
} from "../types";

interface RootViewportProps {
  apiBase: string;
  state: EditorState;
  interactionLocked: boolean;
  onHit: (hit: MeshHit) => void;
  onStroke: (hits: MeshHit[]) => void;
  onError: (message: string) => void;
  onScaleChange: (scale: number) => void;
}

interface CameraTween {
  position: THREE.Vector3;
  target: THREE.Vector3;
}

interface ViewRuntime {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  content: THREE.Group;
  mesh: THREE.Mesh | null;
  geometry: THREE.BufferGeometry | null;
  labels: Int32Array | null;
  surfaceColors: Uint8Array | null;
  lineGroup: THREE.Group;
  relationGroup: THREE.Group;
  lineMaterials: LineMaterial[];
  grid: THREE.GridHelper;
  renderOrigin: THREE.Vector3;
  modelCenter: THREE.Vector3;
  modelRadius: number;
  raycaster: THREE.Raycaster;
  pointer: THREE.Vector2;
  animationFrame: number;
  resizeObserver: ResizeObserver;
  cameraTween: CameraTween | null;
  selectedPatchIndices: Uint32Array | null;
  strokePreview: THREE.Points;
  renderRequested: boolean;
}

const INSERTION_COLOR = rgbToNumber(ROOT_EXPORT_COLORS.higherOrder);
const TIP_COLOR = 0xff5c64;

export function RootViewport({
  apiBase,
  state,
  interactionLocked,
  onHit,
  onStroke,
  onError,
  onScaleChange,
}: RootViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const runtimeRef = useRef<ViewRuntime | null>(null);
  const handledFocusNonceRef = useRef(0);
  const handledPatchFocusNonceRef = useRef(0);
  const selectedRootId = useEditorStore((store) => store.selectedRootId);
  const selectedPatchId = useEditorStore((store) => store.selectedPatchId);
  const activeTool = useEditorStore((store) => store.tool);
  const draftPoints = useEditorStore((store) => store.draftPoints);
  const focusRequest = useEditorStore((store) => store.focusRequest);
  const patchFocusRequest = useEditorStore(
    (store) => store.patchFocusRequest,
  );
  const latestRef = useRef({
    state,
    interactionLocked,
    onHit,
    onStroke,
    onError,
    activeTool,
  });
  useLayoutEffect(() => {
    latestRef.current = {
      state,
      interactionLocked,
      onHit,
      onStroke,
      onError,
      activeTool,
    };
  }, [activeTool, interactionLocked, onError, onHit, onStroke, state]);

  const setHovered = useEditorStore((store) => store.setHovered);
  const setLoadProgress = useEditorStore((store) => store.setLoadProgress);
  const setMeshReady = useEditorStore((store) => store.setMeshReady);
  const setClientGpu = useEditorStore((store) => store.setClientGpu);
  const meshReady = useEditorStore((store) => store.meshReady);

  const meshUrl = apiUrl(apiBase, state.mesh.url);
  const labelUrl = state.mesh.labels_url;
  const renderOrigin = state.mesh.render_origin ?? [0, 0, 0];
  const renderOriginX = renderOrigin[0];
  const renderOriginY = renderOrigin[1];
  const renderOriginZ = renderOrigin[2];
  const rootByLabel = useMemo(
    () => new Map(state.roots.map((root) => [root.numeric_label, root])),
    [state.roots],
  );
  const selectedPatch = useMemo(
    () =>
      state.point_patches.find(
        (patch) => patch.patch_id === selectedPatchId,
      ) ?? null,
    [selectedPatchId, state.point_patches],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.shadowMap.enabled = false;
    renderer.domElement.setAttribute("aria-label", "Interactive 3D root system");
    renderer.domElement.setAttribute("role", "application");
    renderer.domElement.tabIndex = 0;
    container.appendChild(renderer.domElement);

    const gl = renderer.getContext();
    const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
    const clientGpu = debugInfo
      ? String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
      : String(gl.getParameter(gl.RENDERER));
    setClientGpu(clientGpu);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07110f);
    scene.fog = new THREE.FogExp2(0x07110f, 0.00008);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100000);
    camera.up.set(0, 0, 1);
    camera.position.set(8, 5, 8);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.screenSpacePanning = true;
    controls.zoomToCursor = true;
    controls.minDistance = 0.01;
    controls.maxDistance = 100000;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };

    const content = new THREE.Group();
    const lineGroup = new THREE.Group();
    const relationGroup = new THREE.Group();
    const strokePreview = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({
        color: 0xffd067,
        size: 5,
        sizeAttenuation: false,
        depthTest: true,
        depthWrite: false,
      }),
    );
    strokePreview.visible = false;
    strokePreview.renderOrder = 6;
    content.add(lineGroup, relationGroup, strokePreview);
    scene.add(content);

    const hemisphere = new THREE.HemisphereLight(0xd7fff2, 0x14231d, 1.35);
    const rim = new THREE.DirectionalLight(0x6bcbb0, 3);
    rim.position.set(-4, 1, -3);
    const oppositeRim = new THREE.DirectionalLight(0x6bcbb0, 3);
    oppositeRim.position.set(4, -1, 3);
    scene.add(hemisphere, rim, oppositeRim);

    const grid = new THREE.GridHelper(10, 10, 0x2c5148, 0x18302b);
    grid.rotation.x = Math.PI / 2;
    const gridMaterial = grid.material as THREE.LineBasicMaterial;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.34;
    scene.add(grid);

    const raycaster = new THREE.Raycaster();
    (raycaster as THREE.Raycaster & { firstHitOnly?: boolean }).firstHitOnly = true;
    const runtime: ViewRuntime = {
      renderer,
      scene,
      camera,
      controls,
      content,
      mesh: null,
      geometry: null,
      labels: null,
      surfaceColors: null,
      lineGroup,
      relationGroup,
      lineMaterials: [],
      grid,
      renderOrigin: new THREE.Vector3().fromArray(
        latestRef.current.state.mesh.render_origin ?? [0, 0, 0],
      ),
      modelCenter: new THREE.Vector3(),
      modelRadius: 1,
      raycaster,
      pointer: new THREE.Vector2(),
      animationFrame: 0,
      resizeObserver: null as unknown as ResizeObserver,
      cameraTween: null,
      selectedPatchIndices: null,
      strokePreview,
      renderRequested: true,
    };
    runtimeRef.current = runtime;

    const requestRender = () => {
      runtime.renderRequested = true;
    };
    const beginManualCameraControl = () => {
      runtime.cameraTween = null;
      runtime.renderRequested = true;
    };
    controls.addEventListener("change", requestRender);
    controls.addEventListener("start", beginManualCameraControl);

    const resize = () => {
      const width = Math.max(container.clientWidth, 1);
      const height = Math.max(container.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      for (const material of runtime.lineMaterials) {
        material.resolution.set(width, height);
      }
      runtime.renderRequested = true;
    };
    const resizeObserver = new ResizeObserver(resize);
    runtime.resizeObserver = resizeObserver;
    resizeObserver.observe(container);
    resize();

    let previousTime = performance.now();
    const animate = (time: number) => {
      runtime.animationFrame = requestAnimationFrame(animate);
      const delta = Math.min((time - previousTime) / 1000, 0.05);
      previousTime = time;
      const controlsChanged = controls.update(delta);
      if (runtime.cameraTween) {
        camera.position.lerp(runtime.cameraTween.position, 0.14);
        controls.target.lerp(runtime.cameraTween.target, 0.14);
        if (
          camera.position.distanceTo(runtime.cameraTween.position) <
            runtime.modelRadius * 0.0005 &&
          controls.target.distanceTo(runtime.cameraTween.target) <
            runtime.modelRadius * 0.0005
        ) {
          camera.position.copy(runtime.cameraTween.position);
          controls.target.copy(runtime.cameraTween.target);
          runtime.cameraTween = null;
        }
        runtime.renderRequested = true;
      }
      if (controlsChanged || runtime.renderRequested) {
        renderer.render(scene, camera);
        runtime.renderRequested = false;
      }
    };
    runtime.animationFrame = requestAnimationFrame(animate);

    let pointerStart: { x: number; y: number; suppressHit: boolean } | null =
      null;
    let hoverFrame = 0;
    let latestPointerEvent: PointerEvent | null = null;
    let strokePointerId: number | null = null;
    let strokeLastScreen: { x: number; y: number } | null = null;
    let strokeHits: MeshHit[] = [];
    const maximumStrokeSamples = 20_000;

    const findHitAt = (clientX: number, clientY: number): MeshHit | null => {
      if (!runtime.mesh || !runtime.labels || !runtime.geometry) return null;
      const rect = renderer.domElement.getBoundingClientRect();
      runtime.pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      runtime.pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(runtime.pointer, camera);
      const intersection = raycaster.intersectObject(runtime.mesh, false)[0];
      if (!intersection?.face) return null;
      const localPoint = runtime.mesh.worldToLocal(intersection.point.clone());
      const position = runtime.geometry.getAttribute("position");
      const candidates = [
        intersection.face.a,
        intersection.face.b,
        intersection.face.c,
      ];
      let vertexIndex = candidates[0];
      let nearest = Number.POSITIVE_INFINITY;
      for (const candidate of candidates) {
        const dx = position.getX(candidate) - localPoint.x;
        const dy = position.getY(candidate) - localPoint.y;
        const dz = position.getZ(candidate) - localPoint.z;
        const distance = dx * dx + dy * dy + dz * dz;
        if (distance < nearest) {
          nearest = distance;
          vertexIndex = candidate;
        }
      }
      const numericLabel = runtime.labels[vertexIndex];
      const root = latestRef.current.state.roots.find(
        (candidate) => candidate.numeric_label === numericLabel,
      );
      return {
        rootId: root?.root_id ?? null,
        numericLabel,
        position: [
          localPoint.x + runtime.renderOrigin.x,
          localPoint.y + runtime.renderOrigin.y,
          localPoint.z + runtime.renderOrigin.z,
        ],
        vertexIndex,
      };
    };
    const findHit = (event: PointerEvent) =>
      findHitAt(event.clientX, event.clientY);

    const redrawStrokePreview = () => {
      const positions = new Float32Array(strokeHits.length * 3);
      for (let index = 0; index < strokeHits.length; index += 1) {
        const point = strokeHits[index].position;
        positions[index * 3] = point[0] - runtime.renderOrigin.x;
        positions[index * 3 + 1] = point[1] - runtime.renderOrigin.y;
        positions[index * 3 + 2] = point[2] - runtime.renderOrigin.z;
      }
      runtime.strokePreview.geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );
      runtime.strokePreview.visible = strokeHits.length > 0;
      runtime.renderRequested = true;
    };

    const appendStrokeHit = (hit: MeshHit | null) => {
      if (!hit || strokeHits.length >= maximumStrokeSamples) return;
      const previous = strokeHits.at(-1);
      if (previous) {
        const dx = hit.position[0] - previous.position[0];
        const dy = hit.position[1] - previous.position[1];
        const dz = hit.position[2] - previous.position[2];
        const minimumSpacing = runtime.modelRadius * 0.00005;
        if (dx * dx + dy * dy + dz * dz < minimumSpacing * minimumSpacing) {
          return;
        }
      }
      strokeHits.push(hit);
      redrawStrokePreview();
    };

    const sampleStrokePoint = (clientX: number, clientY: number) => {
      const previous = strokeLastScreen;
      const distance = previous
        ? Math.hypot(clientX - previous.x, clientY - previous.y)
        : 0;
      const steps = Math.max(1, Math.ceil(distance / 5));
      for (let step = 1; step <= steps; step += 1) {
        const fraction = step / steps;
        const x = previous
          ? previous.x + (clientX - previous.x) * fraction
          : clientX;
        const y = previous
          ? previous.y + (clientY - previous.y) * fraction
          : clientY;
        appendStrokeHit(findHitAt(x, y));
      }
      strokeLastScreen = { x: clientX, y: clientY };
    };

    const clearStroke = () => {
      strokePointerId = null;
      strokeLastScreen = null;
      strokeHits = [];
      runtime.strokePreview.geometry.deleteAttribute("position");
      runtime.strokePreview.visible = false;
      runtime.renderRequested = true;
    };

    const pointerMove = (event: PointerEvent) => {
      if (event.pointerId === strokePointerId) {
        event.preventDefault();
        event.stopPropagation();
        const samples = event.getCoalescedEvents?.() ?? [event];
        for (const sample of samples) {
          sampleStrokePoint(sample.clientX, sample.clientY);
        }
        setHovered(null);
        return;
      }
      latestPointerEvent = event;
      if (hoverFrame) return;
      hoverFrame = requestAnimationFrame(() => {
        hoverFrame = 0;
        if (!latestPointerEvent || latestRef.current.interactionLocked) {
          setHovered(null);
          return;
        }
        const hit = findHit(latestPointerEvent);
        setHovered(
          hit
            ? {
                ...hit,
                clientX: latestPointerEvent.clientX,
                clientY: latestPointerEvent.clientY,
              }
            : null,
        );
      });
    };
    const pointerDown = (event: PointerEvent) => {
      const assigning = latestRef.current.activeTool === "assign";
      if (
        assigning &&
        event.button === 0 &&
        event.shiftKey
      ) {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (latestRef.current.interactionLocked) return;
        pointerStart = null;
        strokePointerId = event.pointerId;
        strokeLastScreen = null;
        strokeHits = [];
        renderer.domElement.setPointerCapture(event.pointerId);
        sampleStrokePoint(event.clientX, event.clientY);
        setHovered(null);
        return;
      }
      pointerStart = {
        x: event.clientX,
        y: event.clientY,
        suppressHit: assigning,
      };
    };
    const pointerUp = (event: PointerEvent) => {
      if (event.pointerId === strokePointerId) {
        event.preventDefault();
        event.stopImmediatePropagation();
        sampleStrokePoint(event.clientX, event.clientY);
        const completedHits = strokeHits;
        if (renderer.domElement.hasPointerCapture(event.pointerId)) {
          renderer.domElement.releasePointerCapture(event.pointerId);
        }
        clearStroke();
        if (completedHits.length) latestRef.current.onStroke(completedHits);
        return;
      }
      if (!pointerStart || latestRef.current.interactionLocked) return;
      const suppressHit = pointerStart.suppressHit;
      const distance = Math.hypot(
        event.clientX - pointerStart.x,
        event.clientY - pointerStart.y,
      );
      pointerStart = null;
      if (suppressHit || distance > 5 || event.button !== 0) return;
      const hit = findHit(event);
      if (hit) latestRef.current.onHit(hit);
    };
    const pointerCancel = (event: PointerEvent) => {
      if (event.pointerId !== strokePointerId) return;
      event.stopImmediatePropagation();
      clearStroke();
    };
    const lostPointerCapture = (event: PointerEvent) => {
      if (event.pointerId === strokePointerId) clearStroke();
    };
    const pointerLeave = () => {
      if (strokePointerId === null) setHovered(null);
    };
    const contextMenu = (event: MouseEvent) => event.preventDefault();

    renderer.domElement.addEventListener("pointermove", pointerMove);
    renderer.domElement.addEventListener("pointerdown", pointerDown, true);
    renderer.domElement.addEventListener("pointerup", pointerUp, true);
    renderer.domElement.addEventListener("pointercancel", pointerCancel, true);
    renderer.domElement.addEventListener(
      "lostpointercapture",
      lostPointerCapture,
      true,
    );
    renderer.domElement.addEventListener("pointerleave", pointerLeave);
    renderer.domElement.addEventListener("contextmenu", contextMenu);

    return () => {
      cancelAnimationFrame(runtime.animationFrame);
      if (hoverFrame) cancelAnimationFrame(hoverFrame);
      resizeObserver.disconnect();
      controls.removeEventListener("change", requestRender);
      controls.removeEventListener("start", beginManualCameraControl);
      controls.dispose();
      renderer.domElement.removeEventListener("pointermove", pointerMove);
      renderer.domElement.removeEventListener("pointerdown", pointerDown, true);
      renderer.domElement.removeEventListener("pointerup", pointerUp, true);
      renderer.domElement.removeEventListener(
        "pointercancel",
        pointerCancel,
        true,
      );
      renderer.domElement.removeEventListener(
        "lostpointercapture",
        lostPointerCapture,
        true,
      );
      renderer.domElement.removeEventListener("pointerleave", pointerLeave);
      renderer.domElement.removeEventListener("contextmenu", contextMenu);
      disposeGroup(lineGroup);
      disposeGroup(relationGroup);
      runtime.strokePreview.geometry.dispose();
      (runtime.strokePreview.material as THREE.Material).dispose();
      runtime.geometry?.dispose();
      if (runtime.mesh) {
        (runtime.mesh.material as THREE.Material).dispose();
      }
      renderer.dispose();
      renderer.domElement.remove();
      runtimeRef.current = null;
      setHovered(null);
      setMeshReady(false);
    };
  }, [setClientGpu, setHovered, setMeshReady]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    let cancelled = false;
    const parser = new Worker(new URL("../workers/ply.worker.ts", import.meta.url), {
      type: "module",
    });
    let bvhWorker: GenerateMeshBVHWorker | null = null;

    setMeshReady(false);
    setLoadProgress({
      phase: "download",
      progress: 0,
      message: "Preparing the full-resolution mesh",
    });

    parser.onmessage = async (
      event: MessageEvent<
        | ({
            type: "result";
            faceCount: number;
          } & ParsedMesh)
        | {
            type: "progress";
            phase: "download" | "parse" | "shading";
            progress: number;
            message: string;
          }
        | { type: "error"; message: string }
      >,
    ) => {
      if (cancelled) return;
      if (event.data.type === "progress") {
        setLoadProgress(event.data);
        return;
      }
      if (event.data.type === "error") {
        onError(event.data.message);
        setLoadProgress({
          phase: "idle",
          progress: 0,
          message: "Mesh loading failed",
        });
        return;
      }

      parser.terminate();
      const parsed = event.data;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(parsed.positions, 3),
      );
      geometry.setAttribute("normal", new THREE.BufferAttribute(parsed.normals, 3));
      geometry.setIndex(new THREE.BufferAttribute(parsed.indices, 1));
      const surfaceColors = makeSurfaceColors(
        parsed.labels,
        latestRef.current.state.roots,
      );
      geometry.setAttribute(
        "color",
        new THREE.Uint8BufferAttribute(surfaceColors, 3, true),
      );
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();

      runtime.geometry = geometry;
      runtime.labels = parsed.labels;
      runtime.surfaceColors = surfaceColors;
      runtime.renderOrigin.fromArray(
        latestRef.current.state.mesh.render_origin ?? [0, 0, 0],
      );
      runtime.modelCenter.copy(
        geometry.boundingSphere?.center ?? new THREE.Vector3(),
      );
      runtime.modelRadius = Math.max(geometry.boundingSphere?.radius ?? 1, 0.001);
      runtime.content.position.set(0, 0, 0);
      runtime.grid.scale.setScalar(runtime.modelRadius / 5);
      const primary = latestRef.current.state.roots.find(
        (root) => root.root_id === "primary",
      );
      runtime.grid.position.z = primary
        ? primary.insertion_point[2] - runtime.renderOrigin.z
        : 0;
      onScaleChange(runtime.modelRadius);
      const defaultTarget = primary
        ? rootBounds(primary)
            .getCenter(new THREE.Vector3())
            .sub(runtime.renderOrigin)
        : runtime.modelCenter.clone();
      fitCamera(runtime, defaultTarget);

      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.72,
          metalness: 0.02,
          side: THREE.FrontSide,
        }),
      );
      mesh.name = "full-resolution-root-surface";
      (mesh as THREE.Mesh).raycast = acceleratedRaycast;
      runtime.mesh = mesh;

      setLoadProgress({
        phase: "index",
        progress: 0,
        message: "Building the spatial picking index",
      });
      try {
        bvhWorker = new GenerateMeshBVHWorker();
        const boundsTree = await bvhWorker.generate(geometry, {
          strategy: SAH,
          maxLeafTris: 16,
          onProgress: (progress) =>
            setLoadProgress({
              phase: "index",
              progress,
              message: `Building spatial picking index · ${Math.round(progress * 100)}%`,
            }),
        });
        if (cancelled) return;
        (
          geometry as THREE.BufferGeometry & { boundsTree?: MeshBVH }
        ).boundsTree = boundsTree;
      } catch (error) {
        if (cancelled) return;
        latestRef.current.onError(
          `The accelerated picking index could not be created: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      } finally {
        bvhWorker?.dispose();
        bvhWorker = null;
      }
      if (cancelled) return;
      runtime.content.add(mesh);
      runtime.content.children.sort((a) => (a === mesh ? -1 : 1));
      runtime.renderRequested = true;
      setMeshReady(true);
      setLoadProgress({
        phase: "ready",
        progress: 1,
        message: `${parsed.vertexCount.toLocaleString()} vertices · ${parsed.faceCount.toLocaleString()} faces · full resolution`,
      });
    };
    parser.onerror = (event) => {
      if (!cancelled) {
        onError(event.message || "The PLY worker stopped unexpectedly.");
      }
    };
    parser.postMessage({
      type: "load",
      url: meshUrl,
      renderOrigin: [renderOriginX, renderOriginY, renderOriginZ],
    });

    return () => {
      cancelled = true;
      parser.terminate();
      bvhWorker?.dispose();
      if (runtime.mesh) {
        runtime.content.remove(runtime.mesh);
        (runtime.mesh.material as THREE.Material).dispose();
      }
      runtime.geometry?.dispose();
      runtime.mesh = null;
      runtime.geometry = null;
      runtime.labels = null;
      runtime.surfaceColors = null;
    };
    // The immutable geometry URL/count identify the loaded source. Label revisions
    // are fetched by the independent effect below.
  }, [
    meshUrl,
    state.mesh.vertex_count,
    renderOriginX,
    renderOriginY,
    renderOriginZ,
    onError,
    onScaleChange,
    setLoadProgress,
    setMeshReady,
  ]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime?.mesh || !runtime.labels) return;
    const controller = new AbortController();
    fetchLabels(
      apiBase,
      labelUrl,
      state.mesh.vertex_count,
      controller.signal,
    )
      .then((labels) => {
        if (!runtimeRef.current || runtimeRef.current !== runtime) return;
        runtime.labels = labels;
        updateSurfaceColors(
          runtime,
          latestRef.current.state.roots,
        );
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        onError(error instanceof Error ? error.message : String(error));
      });
    return () => controller.abort();
  }, [
    apiBase,
    labelUrl,
    meshReady,
    onError,
    state.mesh.root_label_revision,
    state.mesh.vertex_count,
  ]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime?.mesh || !runtime.labels) return;
    if (!selectedPatch) {
      runtime.selectedPatchIndices = null;
      updateSurfaceColors(runtime, latestRef.current.state.roots);
      return;
    }
    const controller = new AbortController();
    fetchPointPatchIndices(
      apiBase,
      selectedPatch.indices_url,
      selectedPatch.point_count,
      controller.signal,
    )
      .then((indices) => {
        if (!runtimeRef.current || runtimeRef.current !== runtime) return;
        runtime.selectedPatchIndices = indices;
        updateSurfaceColors(runtime, latestRef.current.state.roots);
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        onError(error instanceof Error ? error.message : String(error));
      });
    return () => controller.abort();
  }, [
    apiBase,
    meshReady,
    onError,
    selectedPatch,
    state.mesh.root_label_revision,
  ]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    rebuildCenterlines(
      runtime,
      state.roots,
      selectedPatchId ? null : selectedRootId,
      draftPoints,
      activeTool,
    );
    runtime.renderRequested = true;
  }, [
    activeTool,
    draftPoints,
    selectedPatchId,
    selectedRootId,
    state.roots,
  ]);

  useEffect(() => {
    if (
      !focusRequest ||
      handledFocusNonceRef.current === focusRequest.nonce
    ) {
      return;
    }
    const runtime = runtimeRef.current;
    const root = state.roots.find(
      (candidate) => candidate.root_id === focusRequest.rootId,
    );
    if (!runtime?.mesh || !root) return;
    handledFocusNonceRef.current = focusRequest.nonce;
    focusRoot(runtime, root);
  }, [focusRequest, meshReady, state.roots]);

  useEffect(() => {
    if (
      !patchFocusRequest ||
      handledPatchFocusNonceRef.current === patchFocusRequest.nonce
    ) {
      return;
    }
    const runtime = runtimeRef.current;
    const patch = state.point_patches.find(
      (candidate) => candidate.patch_id === patchFocusRequest.patchId,
    );
    if (!runtime?.mesh || !patch) return;
    handledPatchFocusNonceRef.current = patchFocusRequest.nonce;
    focusPointPatch(runtime, patch);
  }, [meshReady, patchFocusRequest, state.point_patches]);

  return (
    <div className="viewport-shell" ref={containerRef}>
      <div className="viewport-vignette" aria-hidden="true" />
      <div className="axis-key" aria-label="View controls">
        <span><i className="axis-dot axis-x" /> X</span>
        <span><i className="axis-dot axis-y" /> Y</span>
        <span><i className="axis-dot axis-z" /> Z</span>
        <em>
          {activeTool === "assign"
            ? "Shift + left-drag paint · left-drag rotate · wheel zoom"
            : "drag rotate · right-drag pan · wheel zoom"}
        </em>
      </div>
      <div className="resolution-badge">
        <span className="live-dot" />
        {state.mesh.vertex_count.toLocaleString()} vertices
        <strong>FULL</strong>
      </div>
      {rootByLabel.size === 0 ? (
        <div className="viewport-empty">No root paths were found.</div>
      ) : null}
    </div>
  );
}

function makeSurfaceColors(
  labels: Int32Array,
  roots: RootRecord[],
  selectedPatchIndices: Uint32Array | null = null,
): Uint8Array {
  const colors = new Uint8Array(labels.length * 3);
  const rootByLabel = new Map(roots.map((root) => [root.numeric_label, root]));
  for (let vertex = 0; vertex < labels.length; vertex += 1) {
    const root = rootByLabel.get(labels[vertex]);
    let color: readonly [number, number, number];
    if (!root) {
      color =
        labels[vertex] === -2
          ? ROOT_EXPORT_COLORS.uncertain
          : ROOT_EXPORT_COLORS.unassigned;
    } else {
      color = rootOrderRgb(root.root_order);
    }
    const offset = vertex * 3;
    colors[offset] = color[0];
    colors[offset + 1] = color[1];
    colors[offset + 2] = color[2];
  }
  if (selectedPatchIndices) {
    for (const vertex of selectedPatchIndices) {
      if (vertex >= labels.length) continue;
      const offset = vertex * 3;
      colors[offset] = Math.round(colors[offset] * 0.35 + 255 * 0.65);
      colors[offset + 1] = Math.round(
        colors[offset + 1] * 0.35 + 255 * 0.65,
      );
      colors[offset + 2] = Math.round(
        colors[offset + 2] * 0.35 + 255 * 0.65,
      );
    }
  }
  return colors;
}

function updateSurfaceColors(
  runtime: ViewRuntime,
  roots: RootRecord[],
) {
  if (!runtime.geometry || !runtime.labels) return;
  const colors = makeSurfaceColors(
    runtime.labels,
    roots,
    runtime.selectedPatchIndices,
  );
  runtime.surfaceColors = colors;
  runtime.geometry.setAttribute(
    "color",
    new THREE.Uint8BufferAttribute(colors, 3, true),
  );
  runtime.geometry.getAttribute("color").needsUpdate = true;
  runtime.renderRequested = true;
}

function rebuildCenterlines(
  runtime: ViewRuntime,
  roots: RootRecord[],
  selectedRootId: string | null,
  draftPoints: Vec3[],
  activeTool: ToolMode,
) {
  disposeGroup(runtime.lineGroup);
  disposeGroup(runtime.relationGroup);
  runtime.lineMaterials = [];

  let segmentCount = 0;
  for (const root of roots) segmentCount += Math.max(root.polyline.length - 1, 0);
  const positions = new Float32Array(segmentCount * 6);
  const colors = new Uint8Array(segmentCount * 6);
  let cursor = 0;
  for (const root of roots) {
    const color = rootOrderRgb(root.root_order);
    for (let point = 1; point < root.polyline.length; point += 1) {
      const start = root.polyline[point - 1];
      const end = root.polyline[point];
      positions.set(
        [
          start[0] - runtime.renderOrigin.x,
          start[1] - runtime.renderOrigin.y,
          start[2] - runtime.renderOrigin.z,
        ],
        cursor * 3,
      );
      colors.set(color, cursor * 3);
      cursor += 1;
      positions.set(
        [
          end[0] - runtime.renderOrigin.x,
          end[1] - runtime.renderOrigin.y,
          end[2] - runtime.renderOrigin.z,
        ],
        cursor * 3,
      );
      colors.set(color, cursor * 3);
      cursor += 1;
    }
  }
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  lineGeometry.setAttribute(
    "color",
    new THREE.Uint8BufferAttribute(colors, 3, true),
  );
  const baseLines = new THREE.LineSegments(
    lineGeometry,
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: selectedRootId ? 0.2 : 0.58,
      depthWrite: false,
    }),
  );
  baseLines.renderOrder = 3;
  runtime.lineGroup.add(baseLines);

  const selected = roots.find((root) => root.root_id === selectedRootId);
  if (selected) {
    const relations: Array<{
      root: RootRecord;
      color: number;
      width: number;
      opacity: number;
    }> = [{
      root: selected,
      color: rootOrderColorNumber(selected.root_order),
      width: 4.6,
      opacity: 1,
    }];
    const parent = roots.find((root) => root.root_id === selected.parent_id);
    if (parent) {
      relations.push({
        root: parent,
        color: rootOrderColorNumber(parent.root_order),
        width: 3,
        opacity: 0.9,
      });
    }
    for (const childId of selected.children_ids) {
      const child = roots.find((root) => root.root_id === childId);
      if (child) {
        relations.push({
          root: child,
          color: rootOrderColorNumber(child.root_order),
          width: 3,
          opacity: 0.92,
        });
      }
    }
    for (const relation of relations) {
      addWideLine(
        runtime,
        runtime.relationGroup,
        relation.root.polyline,
        relation.color,
        relation.width,
        relation.opacity,
      );
    }

    const markerRadius = markerRadiusForRoot(runtime, selected);
    addMarker(
      runtime,
      runtime.relationGroup,
      selected.insertion_point,
      markerRadius,
      INSERTION_COLOR,
      "Insertion point",
    );
    addMarker(
      runtime,
      runtime.relationGroup,
      selected.tip_point,
      markerRadius,
      TIP_COLOR,
      "Tip point",
    );
  }

  if (draftPoints.length) {
    const selected = roots.find((root) => root.root_id === selectedRootId);
    const previewPoints =
      activeTool === "create" && selected
        ? [
            nearestPolylineNode(selected.polyline, draftPoints[0]),
            ...draftPoints,
          ]
        : draftPoints;
    if (previewPoints.length > 1) {
      addWideLine(
        runtime,
        runtime.relationGroup,
        previewPoints,
        0xffda78,
        3.2,
        0.95,
      );
    }
    const markerRadius = selected
      ? markerRadiusForRoot(runtime, selected) * 0.72
      : Math.max(runtime.modelRadius * 0.003, 0.006);
    if (activeTool === "create" && selected) {
      addMarker(
        runtime,
        runtime.relationGroup,
        previewPoints[0],
        markerRadius,
        INSERTION_COLOR,
        "New root attachment",
      );
    }
    for (const [index, point] of draftPoints.entries()) {
      addMarker(
        runtime,
        runtime.relationGroup,
        point,
        markerRadius,
        index === draftPoints.length - 1 ? 0xfff1bd : 0xffc85e,
        `Draft point ${index + 1}`,
      );
    }
  }
}

function nearestPolylineNode(points: Vec3[], position: Vec3): Vec3 {
  let nearest = points[0];
  let minimumSquared = Number.POSITIVE_INFINITY;
  for (const point of points) {
    const dx = point[0] - position[0];
    const dy = point[1] - position[1];
    const dz = point[2] - position[2];
    const squared = dx * dx + dy * dy + dz * dz;
    if (squared < minimumSquared) {
      nearest = point;
      minimumSquared = squared;
    }
  }
  return nearest;
}

function markerRadiusForRoot(runtime: ViewRuntime, root: RootRecord) {
  const minimum = Math.max(runtime.modelRadius * 0.0005, 0.0025);
  const maximum = Math.max(runtime.modelRadius * 0.009, minimum);
  const rootScale = Math.max(
    root.chord_length ?? 0,
    root.mean_diameter ?? 0,
    minimum,
  );
  return THREE.MathUtils.clamp(rootScale * 0.035, minimum, maximum);
}

function addWideLine(
  runtime: ViewRuntime,
  group: THREE.Group,
  points: Vec3[],
  color: number,
  width: number,
  opacity: number,
) {
  if (points.length < 2) return;
  const geometry = new LineGeometry();
  geometry.setPositions(
    points.flatMap((point) => [
      point[0] - runtime.renderOrigin.x,
      point[1] - runtime.renderOrigin.y,
      point[2] - runtime.renderOrigin.z,
    ]),
  );
  const material = new LineMaterial({
    color,
    linewidth: width,
    transparent: opacity < 1,
    opacity,
    depthTest: true,
    depthWrite: false,
  });
  material.resolution.set(
    runtime.renderer.domElement.clientWidth,
    runtime.renderer.domElement.clientHeight,
  );
  runtime.lineMaterials.push(material);
  const line = new Line2(geometry, material);
  line.computeLineDistances();
  line.renderOrder = 4;
  group.add(line);
}

function addMarker(
  runtime: ViewRuntime,
  group: THREE.Group,
  point: Vec3,
  radius: number,
  color: number,
  name: string,
) {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 14, 10),
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.18,
      roughness: 0.4,
      depthTest: true,
    }),
  );
  marker.position.set(
    point[0] - runtime.renderOrigin.x,
    point[1] - runtime.renderOrigin.y,
    point[2] - runtime.renderOrigin.z,
  );
  marker.name = name;
  marker.renderOrder = 5;
  group.add(marker);
}

function fitCamera(runtime: ViewRuntime, target: THREE.Vector3) {
  runtime.cameraTween = null;
  const distance =
    runtime.modelRadius /
    Math.sin(THREE.MathUtils.degToRad(runtime.camera.fov * 0.5));
  runtime.camera.near = Math.max(runtime.modelRadius / 10000, 0.0001);
  runtime.camera.far = Math.max(runtime.modelRadius * 100, 100);
  runtime.camera.updateProjectionMatrix();
  runtime.controls.target.copy(target);
  runtime.camera.position
    .copy(target)
    .add(new THREE.Vector3(distance * 0.72, distance * 0.42, distance * 0.72));
  runtime.controls.minDistance = runtime.modelRadius * 0.02;
  runtime.controls.maxDistance = runtime.modelRadius * 40;
  runtime.controls.update();
  runtime.renderRequested = true;
}

function focusRoot(runtime: ViewRuntime, root: RootRecord) {
  focusBounds(runtime, rootBounds(root));
}

function focusPointPatch(runtime: ViewRuntime, patch: PointPatchRecord) {
  focusBounds(
    runtime,
    new THREE.Box3(
      new THREE.Vector3().fromArray(patch.bounds.minimum),
      new THREE.Vector3().fromArray(patch.bounds.maximum),
    ),
  );
}

function rootBounds(root: RootRecord): THREE.Box3 {
  const box = new THREE.Box3();
  for (const point of root.polyline) {
    box.expandByPoint(new THREE.Vector3().fromArray(point));
  }
  return box;
}

function focusBounds(runtime: ViewRuntime, box: THREE.Box3) {
  const center = box
    .getCenter(new THREE.Vector3())
    .sub(runtime.renderOrigin);
  const size = Math.max(box.getSize(new THREE.Vector3()).length(), runtime.modelRadius * 0.04);
  const direction = runtime.camera.position
    .clone()
    .sub(runtime.controls.target)
    .normalize();
  const distance = Math.max(size * 1.8, runtime.modelRadius * 0.08);
  runtime.cameraTween = {
    target: center,
    position: center.clone().addScaledVector(direction, distance),
  };
  runtime.renderRequested = true;
}

function disposeGroup(group: THREE.Group) {
  for (const child of [...group.children]) {
    group.remove(child);
    child.traverse((object) => {
      const renderable = object as THREE.Mesh;
      if (renderable.geometry) renderable.geometry.dispose();
      const material = renderable.material;
      if (Array.isArray(material)) {
        for (const item of material) item.dispose();
      } else if (material) {
        material.dispose();
      }
    });
  }
}
