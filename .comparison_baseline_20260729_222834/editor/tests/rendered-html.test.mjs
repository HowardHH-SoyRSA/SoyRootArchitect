import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("produces a static client that the local Python server can serve", async () => {
  const html = await readFile(new URL("dist/client/index.html", root), "utf8");
  assert.match(html, /<title>SoyRoot Studio · 3D Graph Editor<\/title>/);
  assert.match(html, /id="root"/);
  assert.match(html, /assets\/editor-[^"]+\.js/);
  assert.match(html, /assets\/editor-[^"]+\.css/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("ships both workers required for full-resolution interaction", async () => {
  const indexHtml = await readFile(new URL("dist/client/index.html", root), "utf8");
  const entryName = indexHtml.match(/assets\/(editor-[^"]+\.js)/)?.[1];
  assert.ok(entryName, "static entry asset is referenced");
  const entry = await readFile(new URL(`dist/client/assets/${entryName}`, root), "utf8");
  assert.match(entry, /ply\.worker-/);
  assert.match(entry, /generateMeshBVH\.worker-/);
  await access(new URL("app/workers/ply.worker.ts", root));
});

test("keeps the automatic result immutable and exposes every editor operation", async () => {
  const [editor, viewport, page, layout, packageJson] = await Promise.all([
    readFile(new URL("app/components/RootEditor.tsx", root), "utf8"),
    readFile(new URL("app/components/RootViewport.tsx", root), "utf8"),
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/layout.tsx", root), "utf8"),
    readFile(new URL("package.json", root), "utf8"),
  ]);
  for (const operation of [
    "create_root",
    "split_root",
    "merge_roots",
    "assign_points",
    "reconnect_root",
    "reparent_root",
    "delete_root",
    "redraw_root",
    "correct_root_order",
  ]) {
    assert.match(editor, new RegExp(operation));
  }
  assert.match(editor, /Source preserved/);
  assert.match(editor, /Ctrl Shift Z/);
  assert.match(viewport, /powerPreference:\s*"high-performance"/);
  assert.match(viewport, /renderOrigin/);
  assert.match(viewport, /GenerateMeshBVHWorker/);
  assert.match(page, /<RootEditor \/>/);
  assert.match(layout, /SoyRoot Studio/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});

test("single hierarchy selection highlights and frames the complete root", async () => {
  const [editor, tree, viewport, css] = await Promise.all([
    readFile(new URL("app/components/RootEditor.tsx", root), "utf8"),
    readFile(new URL("app/components/RootTree.tsx", root), "utf8"),
    readFile(new URL("app/components/RootViewport.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(editor, /const selectAndFocus = useCallback/);
  assert.match(editor, /setSelectedRootId\(rootId\)/);
  assert.match(editor, /requestFocus\(rootId\)/);
  assert.match(editor, /<RootTree[\s\S]*onSelect=\{selectAndFocus\}/);
  assert.match(tree, /onClick=\{\(\) => onSelect\(root\.root_id\)\}/);
  assert.match(tree, /selectedRootId === root\.root_id && !selectedPatchId/);
  assert.match(viewport, /focusRoot\(runtime, root\)/);
  assert.match(css, /\.tree-row\.selected/);
});

test("creates an undoable root only from an unassigned drawn path", async () => {
  const [editor, chrome, store, details] = await Promise.all([
    readFile(new URL("app/components/RootEditor.tsx", root), "utf8"),
    readFile(new URL("app/components/EditorChrome.tsx", root), "utf8"),
    readFile(new URL("app/store.ts", root), "utf8"),
    readFile(new URL("app/components/RootDetails.tsx", root), "utf8"),
  ]);
  assert.match(chrome, /id: "create"/);
  assert.match(editor, /hit\.numericLabel !== -1/);
  assert.match(editor, /"create_root"/);
  assert.match(editor, /claim_radius: brushRadius/);
  assert.match(editor, /setSelectedRootId\(newRootId\)/);
  assert.match(store, /tool === "redraw" \|\| tool === "create"/);
  assert.match(details, /Path claim radius/);
});

test("preserves the active edit view with equal opposing rim lights", async () => {
  const [editor, viewport, store] = await Promise.all([
    readFile(new URL("app/components/RootEditor.tsx", root), "utf8"),
    readFile(new URL("app/components/RootViewport.tsx", root), "utf8"),
    readFile(new URL("app/store.ts", root), "utf8"),
  ]);
  assert.doesNotMatch(store, /homeViewRequest|requestHomeView/);
  assert.doesNotMatch(editor, /requestHomeView/);
  assert.doesNotMatch(viewport, /homeViewRequest/);
  assert.equal(
    viewport.match(/fitCamera\(runtime, defaultTarget\)/g)?.length,
    1,
  );
  assert.match(viewport, /handledFocusNonceRef\.current === focusRequest\.nonce/);
  assert.match(
    viewport,
    /handledFocusNonceRef\.current = focusRequest\.nonce;\s*focusRoot\(runtime, root\)/,
  );
  assert.doesNotMatch(viewport, /AmbientLight/);
  assert.match(viewport, /new THREE\.HemisphereLight\(0xd7fff2, 0x14231d, 1\.35\)/);
  assert.equal(
    viewport.match(/new THREE\.DirectionalLight\(0x6bcbb0, 3\)/g)?.length,
    2,
  );
  assert.match(viewport, /rim\.position\.set\(-4, 1, -3\)/);
  assert.match(viewport, /oppositeRim\.position\.set\(4, -1, 3\)/);
  assert.match(viewport, /scene\.add\(hemisphere, rim, oppositeRim\)/);
});

test("lists and highlights connected uncertain and unassigned point patches", async () => {
  const [types, api, editor, tree, details, viewport, store, css] =
    await Promise.all([
      readFile(new URL("app/types.ts", root), "utf8"),
      readFile(new URL("app/lib/api.ts", root), "utf8"),
      readFile(new URL("app/components/RootEditor.tsx", root), "utf8"),
      readFile(new URL("app/components/RootTree.tsx", root), "utf8"),
      readFile(new URL("app/components/RootDetails.tsx", root), "utf8"),
      readFile(new URL("app/components/RootViewport.tsx", root), "utf8"),
      readFile(new URL("app/store.ts", root), "utf8"),
      readFile(new URL("app/globals.css", root), "utf8"),
    ]);
  assert.match(types, /interface PointPatchRecord/);
  assert.match(types, /point_patches: PointPatchRecord\[\]/);
  assert.match(api, /fetchPointPatchIndices/);
  assert.match(editor, /pointPatches=\{serverState\.point_patches\}/);
  assert.match(editor, /onSelectPatch=\{selectPatchAndFocus\}/);
  assert.match(tree, /title="Uncertain patches"/);
  assert.match(tree, /title="Unassigned patches"/);
  assert.match(tree, /patch\.point_count\.toLocaleString\(\)/);
  assert.match(details, /function PointPatchDetails/);
  assert.match(viewport, /runtime\.selectedPatchIndices = indices/);
  assert.match(viewport, /focusPointPatch\(runtime, patch\)/);
  assert.match(store, /samePatchRevision/);
  assert.match(css, /\.patch-row\.selected/);
});

test("paints a continuous drag as one undoable assignment operation", async () => {
  const [editor, viewport, chrome, details] = await Promise.all([
    readFile(new URL("app/components/RootEditor.tsx", root), "utf8"),
    readFile(new URL("app/components/RootViewport.tsx", root), "utf8"),
    readFile(new URL("app/components/EditorChrome.tsx", root), "utf8"),
    readFile(new URL("app/components/RootDetails.tsx", root), "utf8"),
  ]);
  assert.match(editor, /const handleBrushStroke = useCallback/);
  assert.match(editor, /positions: hits\.map\(\(hit\) => hit\.position\)/);
  assert.match(editor, /onStroke=\{handleBrushStroke\}/);
  assert.match(viewport, /event\.getCoalescedEvents\?\.\(\) \?\? \[event\]/);
  assert.match(viewport, /Math\.ceil\(distance \/ 5\)/);
  assert.match(viewport, /setPointerCapture\(event\.pointerId\)/);
  assert.match(viewport, /event\.stopImmediatePropagation\(\)/);
  assert.match(viewport, /event\.button === 0 &&\s*event\.shiftKey/);
  assert.match(viewport, /suppressHit: assigning/);
  assert.doesNotMatch(viewport, /!event\.altKey/);
  assert.match(
    viewport,
    /addEventListener\("pointerdown", pointerDown, true\)/,
  );
  assert.match(viewport, /latestRef\.current\.onStroke\(completedHits\)/);
  assert.match(chrome, /Shift \+ left-drag to paint one continuous/);
  assert.match(chrome, /Left-drag rotates/);
  assert.match(details, /full stroke is stored as one\s*undoable operation/);
  assert.match(details, /left-drag rotates normally/);
});

test("uses the primary-root center as the default rotation target", async () => {
  const viewport = await readFile(
    new URL("app/components/RootViewport.tsx", root),
    "utf8",
  );
  assert.match(
    viewport,
    /const defaultTarget = primary[\s\S]*rootBounds\(primary\)[\s\S]*getCenter\(new THREE\.Vector3\(\)\)[\s\S]*sub\(runtime\.renderOrigin\)[\s\S]*runtime\.modelCenter\.clone\(\)/,
  );
  assert.match(viewport, /fitCamera\(runtime, defaultTarget\)/);
  assert.match(
    viewport,
    /function fitCamera\(runtime: ViewRuntime, target: THREE\.Vector3\)/,
  );
  assert.match(viewport, /runtime\.controls\.target\.copy\(target\)/);
  assert.match(viewport, /runtime\.camera\.position\s*\.copy\(target\)/);
});

test("inherits the SoyRootBio exported root color contract", async () => {
  const [palette, viewport, tree, details, legend] = await Promise.all([
    readFile(new URL("app/lib/rootColors.ts", root), "utf8"),
    readFile(new URL("app/components/RootViewport.tsx", root), "utf8"),
    readFile(new URL("app/components/RootTree.tsx", root), "utf8"),
    readFile(new URL("app/components/RootDetails.tsx", root), "utf8"),
    readFile(new URL("app/components/RootColorLegend.tsx", root), "utf8"),
  ]);
  for (const expected of [
    /unassigned:\s*\[140,\s*140,\s*140\]/,
    /uncertain:\s*\[250,\s*122,\s*13\]/,
    /primary:\s*\[13,\s*59,\s*224\]/,
    /order1:\s*\[255,\s*0,\s*255\]/,
    /order2:\s*\[0,\s*158,\s*115\]/,
    /order3:\s*\[140,\s*51,\s*209\]/,
    /higherOrder:\s*\[242,\s*166,\s*20\]/,
  ]) {
    assert.match(palette, expected);
  }
  assert.match(viewport, /rootOrderRgb\(root\.root_order\)/);
  assert.match(tree, /rootOrderCssColor\(root\.root_order\)/);
  assert.match(details, /rootOrderCssColor\(root\.root_order\)/);
  assert.match(legend, /className=\{`root-color-legend/);
  assert.match(legend, /color\.toUpperCase\(\)/);
  assert.doesNotMatch(viewport, /\[88,\s*188,\s*156\]/);
});

test("redraws every wheel zoom and cancels automated focus on manual control", async () => {
  const viewport = await readFile(
    new URL("app/components/RootViewport.tsx", root),
    "utf8",
  );
  assert.match(
    viewport,
    /controls\.addEventListener\("change", requestRender\)/,
  );
  assert.match(
    viewport,
    /controls\.addEventListener\("start", beginManualCameraControl\)/,
  );
  assert.match(
    viewport,
    /controls\.removeEventListener\("change", requestRender\)/,
  );
  assert.match(
    viewport,
    /controls\.removeEventListener\("start", beginManualCameraControl\)/,
  );
});
