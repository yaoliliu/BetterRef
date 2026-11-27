import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const style = document.createElement("link");
style.rel = "stylesheet";
style.href = new URL("./h3_ref_editor.css", import.meta.url);
document.head.append(style);

const widget = (node, name) => node?.widgets?.find((w) => w.name === name);
const photo = (src) =>
  new Promise((resolve, reject) => {
    if (!src) return resolve(null);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("图片载入失败，请重新打开或运行工作流。 Image unavailable. Reopen the editor or run the workflow."));
    img.src = src;
  });
function element(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text) el.textContent = text;
  return el;
}
function button(text, action, cls = "") {
  const b = element("button", cls, text);
  b.type = "button";
  b.onclick = action;
  return b;
}
function toolButton(label, english, action) {
  const control = button(label, action);
  control.append(element("small", "br-en", english));
  control.setAttribute("aria-label", label + " · " + english);
  return control;
}
function downstream(node, type, seen = new Set()) {
  if (!node || seen.has(node)) return;
  if (node.type === type) return node;
  seen.add(node);
  for (const [i] of (node.outputs ?? []).entries()) {
    for (const next of node.getOutputNodes(i) ?? []) {
      const found = downstream(next, type, seen);
      if (found) return found;
    }
  }
}
function referenceSource(node) {
  const index = node.findInputSlot("image");
  const src = node.getInputNode(index);
  const link = node.getInputLink(index);
  if (src?.type === "BetterRefH3Examples") {
    return api.apiURL("/betterref/v1/example/" + (link.origin_slot === 0 ? "baoyu" : "daiyu"));
  }
  if (src?.type === "LoadImage") {
    const filename = widget(src, "image")?.value;
    if (typeof filename === "string")
      return api.apiURL("/view?" + new URLSearchParams({ filename, type: "input" }));
  }
  return src?.imgs?.[0]?.src;
}
function imageCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}
function maskPng(layer) {
  const flat = imageCanvas(layer.width, layer.height);
  const ctx = flat.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, flat.width, flat.height);
  ctx.drawImage(layer, 0, 0);
  return flat.toDataURL("image/png");
}
async function makeLayer(width, height, data) {
  const c = imageCanvas(width, height);
  if (data) {
    const ctx = c.getContext("2d", { willReadFrequently: true });
    const im = await photo(data);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(im, 0, 0, width, height);
    const pixels = ctx.getImageData(0, 0, width, height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      pixels.data[i + 3] = pixels.data[i] >= 128 ? 255 : 0;
      pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = 255;
    }
    ctx.putImageData(pixels, 0, 0);
  }
  return c;
}
function editorOutput(node) {
  const qwen = node.type === "BetterRefQwen21Role";
  const scene = downstream(node, qwen ? "BetterRefQwen21Generate" : "BetterRefH3Scene");
  const locate = qwen ? scene : downstream(scene, "BetterRefH3Locate");
  const last = app.nodeOutputs?.[locate?.id]?.betterref_editor?.[0];
  const width = Number(widget(scene, "width")?.value) || 1536;
  const height = Number(widget(scene, "height")?.value) || 1024;
  const matches = last && last.width === width && last.height === height &&
    String(last.seed) === String(widget(locate, "seed")?.value);
  const role = last?.roles?.[String(node.id)];
  return {
    width, height, role,
    targetAuto: matches ? role?.targetAuto : null,
    baseline: matches && (qwen || widget(locate, "compare")?.value) ? last.baseline : null,
  };
}

function tintMask(canvas, layer) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(layer, 0, 0);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = "#69d9b4";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "source-over";
}

function addMaskPreview(node, stored, side) {
  const isTarget = side === "target";
  const panel = element("div", "br-ref-preview");
  const original = new Image();
  original.alt = isTarget ? "目标原图 · Original target" : "参考原图 · Original reference";
  const overlay = imageCanvas(1, 1);
  overlay.setAttribute("aria-label", isTarget ? "目标图叠加遮罩 · Target mask overlay" : "参考图叠加遮罩 · Reference mask overlay");
  const blank = element("span", "br-blank-frame", "空白画布 · Blank canvas");
  const status = element("span", "br-ref-status");
  for (const [label, view] of [["原图 · Original", original], ["遮罩叠加 · Overlay", overlay]]) {
    const figure = element("figure");
    figure.append(view, element("figcaption", null, label));
    if (view === original) figure.append(blank);
    panel.append(figure);
  }
  panel.append(status);
  let previous, version = 0, lastRole, sourceAtRun, imageLinkAtRun, externalAtRun, editsText, manual;
  function refresh() {
    if (!node.graph) return;
    const {role, width, height, baseline, targetAuto} = editorOutput(node);
    const source = referenceSource(node);
    const imageLink = node.getInputLink(node.findInputSlot("image"));
    const external = node.getInputLink(node.findInputSlot(side + "_mask"));
    const externalKey = external?.id;
    if (role !== lastRole) {
      lastRole = role;
      sourceAtRun = source;
      imageLinkAtRun = imageLink?.id;
      externalAtRun = externalKey;
    }
    if (editsText !== stored.value) {
      editsText = stored.value;
      try {
        manual = JSON.parse(editsText || "{}")[side];
      } catch {
        manual = null;
      }
    }
    const current = source === sourceAtRun && imageLink?.id === imageLinkAtRun ? role : null;
    const image = isTarget ? baseline : source || current?.reference;
    const mask = external ? (externalKey === externalAtRun && current?.[side + "External"]) :
      manual || (isTarget ? current && targetAuto : current?.referenceAuto);
    const label = mask ? (external ? "外部 · External" : manual ? "手绘 · Painted" : "自动 · Auto") :
      "待运行 · Run to preview";
    if (previous?.image === image && previous?.mask === mask && previous?.label === label &&
        previous.width === width && previous.height === height) return;
    previous = {image, mask, label, width, height};
    const ticket = ++version;
    status.textContent = label;
    original.hidden = !image;
    blank.hidden = !!image;
    if (image) original.src = image;
    else original.removeAttribute("src");
    overlay.hidden = true;
    if (!image && !isTarget) return;
    (async () => {
      const bg = await photo(image);
      const frameWidth = bg?.naturalWidth || width, frameHeight = bg?.naturalHeight || height;
      const scale = Math.min(1, 480 / Math.max(frameWidth, frameHeight));
      const w = Math.max(1, Math.round(frameWidth * scale)), h = Math.max(1, Math.round(frameHeight * scale));
      const layer = mask && await makeLayer(w, h, mask);
      if (ticket !== version) return;
      overlay.width = w; overlay.height = h;
      const ctx = overlay.getContext("2d");
      if (bg) ctx.drawImage(bg, 0, 0, w, h);
      else { ctx.fillStyle = "#27342e"; ctx.fillRect(0, 0, w, h); }
      if (layer) {
        const tinted = imageCanvas(w, h);
        tintMask(tinted, layer);
        ctx.globalAlpha = 0.5;
        ctx.drawImage(tinted, 0, 0);
        ctx.globalAlpha = 1;
      }
      overlay.hidden = false;
    })().catch(() => {
      if (ticket === version) status.textContent = "无法预览 · Preview unavailable";
    });
  }
  const preview = node.addDOMWidget(side + "_overlay", "betterref-reference-preview", panel, {
    serialize: false, hideOnZoom: false, getMinHeight: () => 208, getMaxHeight: () => 208, onDraw: refresh,
  });
  preview.serialize = false;
  return { preview, refresh };
}

async function openEditor(node, firstSide) {
  if (document.querySelector(".br-editor")) return;
  const stored = widget(node, "mask_edits");
  const drafts = { ...JSON.parse(stored.value || "{}") };
  const { width, height, role, baseline, targetAuto } = editorOutput(node);

  const dialog = element("dialog", "br-editor");
  dialog.setAttribute("aria-label", "BetterRef 遮罩编辑器 · Mask Editor");
  const top = element("header", "br-top");
  const brand = element("div", "br-brand", "BetterRef");
  const title = element("h2", null, "遮罩编辑器 · Mask Editor");
  const close = button("×", () => dialog.close(), "br-close");
  close.setAttribute("aria-label", "取消并关闭 · Close without saving");
  top.append(brand, title, close);
  const workspace = element("div", "br-workspace");
  const stage = element("div", "br-stage");
  const viewport = element("div", "br-viewport");
  const display = imageCanvas(1, 1);
  display.setAttribute("aria-label", "遮罩绘制画布 · Mask canvas");
  const cursor = element("i", "br-cursor");
  viewport.append(display, cursor);
  stage.append(viewport);
  const controls = element("aside", "br-tools");
  const label = element("div", "br-role", widget(node, "name")?.value || "角色 · Role");
  const tabs = element("div", "br-tabs");
  const refTab = toolButton("参考图", "Reference", () => switchSide("reference"));
  const tarTab = toolButton("目标图", "Target", () => switchSide("target"));
  tabs.append(refTab, tarTab);
  const group = element("div", "br-toolpair");
  const brush = toolButton("画笔", "Brush", () => changeTool("paint"));
  const erase = toolButton("橡皮", "Eraser", () => changeTool("erase"));
  group.append(brush, erase);
  const msg = element("div", "br-state");
  const footer = element("footer", "br-footer");
  const hist = element("div", "br-history");
  const undo = button("撤销 · Undo", () => moveHistory(-1));
  const redo = button("重做 · Redo", () => moveHistory(1));
  hist.append(undo, redo);
  const save = button(
    "保存遮罩 · Save mask",
    () => {
      stored.value = JSON.stringify(drafts);
      stored.callback?.(stored.value);
      node.refreshMaskPreviews?.();
      node.graph?.change();
      node.setDirtyCanvas(true, true);
      dialog.close();
    },
    "br-save",
  );
  footer.append(
    hist,
    button("取消 · Cancel", () => dialog.close()),
    save,
  );
  let side,
    state,
    tool = "paint",
    radiusPct = 2.2,
    opacity = 0.5,
    zoom = 1,
    active = null,
    generation = 0;
  const states = {};
  function slider(name, min, max, step, value, format, change) {
    const row = element("label", "br-slider"),
      val = element("output", null, format(value));
    const input = document.createElement("input");
    input.type = "range";
    Object.assign(input, { min, max, step, value });
    input.setAttribute("aria-label", name);
    input.oninput = () => {
      val.textContent = format(+input.value);
      change(+input.value);
    };
    row.append(element("span", null, name), val, input);
    return row;
  }
  const reset = button(
    "恢复自动定位 · Restore auto mask",
    async () => {
      if (!state || state.locked) return;
      await updateLayer(state.auto, true);
    },
    "br-reset",
  );
  const clear = button(
    "清空当前遮罩 · Clear mask",
    async () => {
      if (state && !state.locked) await updateLayer(null, false);
    },
    "br-reset",
  );
  controls.append(
    label,
    tabs,
    element("h3", null, "绘制工具 · Tools"),
    group,
    slider(
      "笔刷直径 · Brush size",
      0.3,
      15,
      0.1,
      radiusPct,
      (n) => n.toFixed(1) + "%",
      (n) => (radiusPct = n),
    ),
    slider(
      "遮罩透明度 · Opacity",
      0,
      1,
      0.05,
      opacity,
      (n) => Math.round(n * 100) + "%",
      (n) => {
        opacity = n;
        render();
      },
    ),
    slider(
      "画布缩放 · Zoom",
      1,
      3,
      0.25,
      zoom,
      (n) => n + "×",
      (n) => {
        zoom = n;
        fit();
      },
    ),
    reset,
    clear,
    msg,
  );
  workspace.append(stage, controls);
  dialog.append(top, workspace, footer);
  document.body.append(dialog);
  const resize = new ResizeObserver(fit);
  resize.observe(stage);
  dialog.addEventListener(
    "close",
    () => {
      generation++;
      resize.disconnect();
      dialog.remove();
    },
    { once: true },
  );
  dialog.showModal();

  function fit() {
    if (!state) return;
    const scale =
      Math.min((stage.clientWidth - 36) / display.width, (stage.clientHeight - 36) / display.height) *
      zoom;
    viewport.style.width = Math.max(1, display.width * scale) + "px";
    viewport.style.height = Math.max(1, display.height * scale) + "px";
  }
  function render() {
    if (!state) return;
    const ctx = display.getContext("2d"),
      { layer, bg } = state;
    ctx.clearRect(0, 0, display.width, display.height);
    if (bg) ctx.drawImage(bg, 0, 0, display.width, display.height);
    tintMask(state.overlay, layer);
    ctx.globalAlpha = opacity;
    ctx.drawImage(state.overlay, 0, 0);
    ctx.globalAlpha = 1;
    undo.disabled = !state.index || state.locked;
    redo.disabled = state.index >= state.history.length - 1 || state.locked;
    const source = state.locked
      ? state.external
        ? "外部遮罩 · Read-only external mask"
        : "外部遮罩 · Run to preview"
      : drafts[side]
        ? "手绘遮罩 · Painted mask"
        : state.auto
          ? "自动定位 · Auto mask"
          : "尚无自动遮罩 · No auto mask yet";
    msg.textContent = source + (bg ? "" : " · 空白底图 / Blank canvas");
    reset.disabled = clear.disabled = brush.disabled = erase.disabled = state.locked;
  }
  function changeTool(value) {
    tool = value;
    brush.setAttribute("aria-pressed", value === "paint");
    erase.setAttribute("aria-pressed", value === "erase");
  }
  function record(manual) {
    const value = manual ? maskPng(state.layer) : null;
    if (value) drafts[side] = value;
    else delete drafts[side];
    state.history.splice(state.index + 1);
    state.history.push(value);
    if (state.history.length > 25) state.history.shift();
    state.index = state.history.length - 1;
    render();
  }
  async function updateLayer(data, automatic) {
    const current = state;
    const layer = await makeLayer(current.layer.width, current.layer.height, data);
    if (state !== current) return;
    current.layer = layer;
    record(!automatic);
  }
  async function moveHistory(delta) {
    if (!state || state.locked) return;
    const current = state,
      index = current.index + delta;
    if (index < 0 || index >= current.history.length) return;
    const value = current.history[index];
    const layer = await makeLayer(current.layer.width, current.layer.height, value ?? current.auto);
    if (state !== current) return;
    current.index = index;
    current.layer = layer;
    if (value) drafts[side] = value;
    else delete drafts[side];
    render();
  }
  async function switchSide(next) {
    const ticket = ++generation;
    active = null;
    side = next;
    state = null;
    cursor.hidden = true;
    refTab.setAttribute("aria-pressed", next === "reference");
    tarTab.setAttribute("aria-pressed", next === "target");
    msg.textContent = "正在载入底图 · Loading image…";
    display.getContext("2d").clearRect(0, 0, display.width, display.height);
    try {
      if (!states[next]) {
        const isRef = next === "reference";
        const src = isRef
          ? referenceSource(node) || role?.reference
          : baseline;
        const bg = await photo(src);
        const auto = isRef ? role?.referenceAuto : targetAuto;
        const fallback = isRef && !bg && drafts[next] ? await photo(drafts[next]) : null;
        const w = bg?.naturalWidth || fallback?.naturalWidth || (isRef ? 768 : width),
          h = bg?.naturalHeight || fallback?.naturalHeight || (isRef ? 768 : height);
        const scale = Math.min(1, 1024 / Math.max(w, h)),
          dw = Math.max(1, Math.round(w * scale)),
          dh = Math.max(1, Math.round(h * scale));
        const locked = node.inputs?.some((i) => i.name === next + "_mask" && i.link != null);
        const external = role?.[next + "External"];
        const layer = await makeLayer(dw, dh, locked ? external : (drafts[next] ?? auto));
        states[next] = {
          bg,
          auto,
          layer,
          overlay: imageCanvas(dw, dh),
          history: [drafts[next] ?? null],
          index: 0,
          locked,
          external,
        };
      }
      if (ticket !== generation) return;
      state = states[next];
      display.width = state.layer.width;
      display.height = state.layer.height;
      fit();
      render();
    } catch (err) {
      if (ticket === generation) msg.textContent = err.message;
    }
  }
  function point(e) {
    const box = display.getBoundingClientRect();
    return [
      ((e.clientX - box.left) * display.width) / box.width,
      ((e.clientY - box.top) * display.height) / box.height,
    ];
  }
  function stroke(start, end) {
    const ctx = state.layer.getContext("2d"),
      d = Math.max(1, (radiusPct * display.width) / 100);
    ctx.globalCompositeOperation = tool === "erase" ? "destination-out" : "source-over";
    ctx.fillStyle = ctx.strokeStyle = "white";
    ctx.lineCap = "round";
    ctx.lineWidth = d;
    ctx.beginPath();
    ctx.moveTo(...start);
    ctx.lineTo(...end);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(end[0], end[1], d / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
    render();
  }
  display.onpointerdown = (e) => {
    if (!state || state.locked || e.button !== 0 || active) return;
    e.preventDefault();
    display.setPointerCapture(e.pointerId);
    active = { id: e.pointerId, point: point(e) };
    stroke(active.point, active.point);
  };
  display.onpointermove = (e) => {
    if (!state) return;
    const box = display.getBoundingClientRect();
    cursor.hidden = false;
    Object.assign(cursor.style, {
      left: e.clientX - box.left + "px",
      top: e.clientY - box.top + "px",
      width: (box.width * radiusPct) / 100 + "px",
      height: (box.width * radiusPct) / 100 + "px",
    });
    if (active?.id === e.pointerId) {
      let end = point(e);
      stroke(active.point, end);
      active.point = end;
    }
  };
  display.onpointerleave = () => {
    cursor.hidden = true;
  };
  display.onpointerup = display.onpointercancel = (e) => {
    if (active?.id === e.pointerId) {
      active = null;
      record(true);
    }
  };
  dialog.onkeydown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      moveHistory(e.shiftKey ? 1 : -1);
    } else if (e.key.toLowerCase() === "b") changeTool("paint");
    else if (e.key.toLowerCase() === "e") changeTool("erase");
  };
  changeTool("paint");
  await switchSide(firstSide);
}
function addEditButton(node, side, chinese, english) {
  const launch = button("", () => openEditor(node, side), "br-mask-launch");
  launch.setAttribute("aria-haspopup", "dialog");
  launch.setAttribute("aria-label", chinese + " · " + english);
  const icon = element("span", "br-mask-icon");
  icon.setAttribute("aria-hidden", "true");
  icon.innerHTML = '<svg viewBox="0 0 24 24" width="25" height="25" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.5 21 4a1.4 1.4 0 0 0-2-2L8.5 11.5Z"/><path d="M8.5 13c-4 0-3 4.5-6 6.5 5 1.3 9-.4 9-4"/></svg>';
  const arrow = element("span", "br-mask-arrow", "↗");
  arrow.setAttribute("aria-hidden", "true");
  launch.append(icon, element("strong", null, chinese), element("span", "br-mask-english", english), arrow);
  const edit = node.addDOMWidget(side === "reference" ? "编辑遮罩" : "编辑目标遮罩", "betterref-mask-button", launch, {
    serialize: false, hideOnZoom: false,
    getMinHeight: () => 64, getMaxHeight: () => 64,
  });
  edit.serialize = false;
  return edit;
}

function migrateQwenWidgets(graph) {
  for (const node of graph.nodes ?? []) {
    const values = node.widgets_values;
    if (node.type !== "BetterRefQwen21Generate" || !Array.isArray(values)) continue;
    if (values[5] === "Picture 1 left, Picture 2 right" || values[5] === "Picture 2 left, Picture 1 right") {
      node.widgets_values = [...values.slice(0, 5), ...values.slice(6, 9)];
      node.inputs = node.inputs?.filter((input) => input.name !== "mask_mode");
    }
  }
  for (const subgraph of graph.definitions?.subgraphs ?? []) migrateQwenWidgets(subgraph);
}

app.registerExtension({
  name: "BetterRef.H3RoleEditor",
  beforeConfigureGraph(graph) {
    migrateQwenWidgets(graph);
  },
  afterConfigureGraph() {
    for (const node of app.graph._nodes) node.refreshMaskPreviews?.();
  },
  async beforeRegisterNodeDef(type, data) {
    if (["BetterRefH3Locate", "BetterRefQwen21Generate"].includes(data.name)) {
      const previous = type.prototype.onExecuted;
      type.prototype.onExecuted = function (output) {
        previous?.call(this, output);
        for (const id of Object.keys(output?.betterref_editor?.[0]?.roles ?? {})) {
          this.graph?.getNodeById(id)?.refreshMaskPreviews?.();
        }
      };
    }
    if (!["BetterRefH3Role", "BetterRefQwen21Role"].includes(data.name)) return;
    const previous = type.prototype.onNodeCreated;
    type.prototype.onNodeCreated = function (...args) {
      previous?.apply(this, args);
      for (const [field, label] of [
        ["target_phrase", "目标图描述 · Target phrase"],
        ["reference_phrase", "参考图描述 · Reference phrase"],
      ]) {
        const input = widget(this, field)?.element;
        if (input) {
          input.dataset.betterrefPhrase = field;
          input.setAttribute("aria-label", label);
        }
      }
      const stored = widget(this, "mask_edits");
      if (!stored) return;
      stored.type = "hidden";
      stored.computeSize = () => [0, -4];
      const reference = addMaskPreview(this, stored, "reference");
      const target = addMaskPreview(this, stored, "target");
      this.refreshMaskPreviews = () => { reference.refresh(); target.refresh(); };
      const controls = [
        addEditButton(this, "reference", "编辑参考图遮罩", "Edit reference mask"), reference.preview,
        addEditButton(this, "target", "编辑目标图遮罩", "Edit target mask"), target.preview,
      ];
      for (const control of controls) this.widgets.splice(this.widgets.indexOf(control), 1);
      this.widgets.unshift(...controls);
      this.setSize([Math.max(this.size[0], 475), Math.max(this.size[1], 937)]);
    };
  },
});
