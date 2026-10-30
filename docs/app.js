// ROAD//SCAN front end — Copyright (c) 2026 Rishabh Aryan — MIT License
(() => {
  const S = 640, COLORS = ["#ff3b5c", "#00e5ff", "#b6ff3b"];
  const SAMPLES = ["samples/sample1.jpg", "samples/sample2.jpg", "samples/sample3.jpg", "samples/sample4.jpg", "samples/sample5.jpg"];
  const $ = id => document.getElementById(id);
  const drop = $("drop"), file = $("file"), view = $("view"), canvas = $("canvas"), ctx = canvas.getContext("2d");
  let session = null, lastImg = null, lastOut = null, lastLb = null, busy = false;

  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.InferenceSession.create("model/best.onnx", { executionProviders: ["wasm"] })
    .then(s => { session = s; setStatus("Model ready · YOLO11n", "ok"); })
    .catch(e => { console.error(e); setStatus("Model failed to load", "err"); });

  function setStatus(t, cls) { $("status").textContent = t; $("dot").className = "dot " + (cls || ""); }

  SAMPLES.forEach(src => {
    const b = document.createElement("button");
    b.style.backgroundImage = `url(${src})`; b.title = "Scan this sample"; b.setAttribute("aria-label", "Sample road image");
    b.onclick = e => { e.stopPropagation(); load(src); };
    $("sampleRow").appendChild(b);
  });

  drop.onclick = () => file.click();
  drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") file.click(); };
  file.onchange = () => file.files[0] && load(URL.createObjectURL(file.files[0]));
  ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f && f.type.startsWith("image/")) load(URL.createObjectURL(f)); });
  $("conf").oninput = e => { $("confOut").textContent = (+e.target.value).toFixed(2); if (lastOut) render(); };

  function load(src) {
    if (busy) return;
    const img = new Image();
    img.onload = () => run(img);
    img.src = src;
  }

  async function run(img) {
    if (!session) { setStatus("Model still loading…"); return; }
    busy = true; lastImg = img;
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    $("hint").classList.add("hidden"); view.classList.remove("hidden"); drop.classList.add("scanning");
    setStatus("Scanning…");
    const t0 = performance.now();
    const { tensor, lb } = preprocess(img);
    const res = await session.run({ [session.inputNames[0]]: tensor });
    const out = res[session.outputNames[0]];
    const ms = performance.now() - t0;
    await new Promise(r => setTimeout(r, Math.max(0, 900 - ms)));   // let the laser finish a sweep
    lastOut = out; lastLb = lb; busy = false;
    drop.classList.remove("scanning");
    setStatus("Model ready · YOLO11n", "ok");
    $("timing").textContent = `Inference ${ms.toFixed(0)} ms · ${img.naturalWidth}×${img.naturalHeight} → 640 letterbox · WebAssembly`;
    render(true);
  }

  function preprocess(img) {
    const W = img.naturalWidth, H = img.naturalHeight, scale = Math.min(S / W, S / H);
    const nw = Math.round(W * scale), nh = Math.round(H * scale), padX = (S - nw) / 2, padY = (S - nh) / 2;
    const c = document.createElement("canvas"); c.width = S; c.height = S;
    const g = c.getContext("2d"); g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, S, S);
    g.drawImage(img, padX, padY, nw, nh);
    const px = g.getImageData(0, 0, S, S).data, f = new Float32Array(3 * S * S);
    for (let i = 0, p = 0; i < S * S; i++, p += 4) { f[i] = px[p] / 255; f[S * S + i] = px[p + 1] / 255; f[2 * S * S + i] = px[p + 2] / 255; }
    return { tensor: new ort.Tensor("float32", f, [1, 3, S, S]), lb: { scale, padX, padY } };
  }

  function render(animate) {
    const W = lastImg.naturalWidth, H = lastImg.naturalHeight, [, ch, n] = lastOut.dims;
    const dets = RoadDetect.decode(lastOut.data, n, ch - 4, lastLb, +$("conf").value, 0.7, W, H);
    ctx.drawImage(lastImg, 0, 0);
    const lw = Math.max(2, Math.round(Math.min(W, H) / 220)), fs = Math.max(12, Math.round(Math.min(W, H) / 32));
    ctx.font = `bold ${fs}px ui-monospace, Menlo, monospace`;
    dets.forEach(d => {
      const col = COLORS[d.cls], w = d.x2 - d.x1, h = d.y2 - d.y1;
      ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = 16; ctx.strokeStyle = col; ctx.lineWidth = lw;
      ctx.strokeRect(d.x1, d.y1, w, h); ctx.restore();
      ctx.fillStyle = col + "22"; ctx.fillRect(d.x1, d.y1, w, h);
      const k = Math.min(w, h) * 0.25; ctx.strokeStyle = col; ctx.lineWidth = lw * 2;
      [[d.x1, d.y1, 1, 1], [d.x2, d.y1, -1, 1], [d.x1, d.y2, 1, -1], [d.x2, d.y2, -1, -1]].forEach(([x, y, sx, sy]) => {
        ctx.beginPath(); ctx.moveTo(x + sx * k, y); ctx.lineTo(x, y); ctx.lineTo(x, y + sy * k); ctx.stroke(); });
      const t = `${RoadDetect.CLASSES[d.cls].toUpperCase()} ${(d.score * 100).toFixed(0)}%`, tw = ctx.measureText(t).width + 10;
      const ty = d.y1 - fs - 6 < 0 ? d.y1 : d.y1 - fs - 6;
      ctx.fillStyle = col; ctx.fillRect(d.x1, ty, tw, fs + 6);
      ctx.fillStyle = "#0b0b0b"; ctx.fillText(t, d.x1 + 5, ty + fs);
    });
    const counts = [0, 0, 0]; dets.forEach(d => counts[d.cls]++);
    document.querySelectorAll(".cnt b").forEach(b => countTo(b, counts[+b.dataset.k], animate));
    const score = RoadDetect.health(dets, W, H);
    countTo($("score"), score, animate);
    const ring = $("ring"); ring.style.strokeDashoffset = 314.16 * (1 - score / 100);
    const col = score >= 80 ? "#b6ff3b" : score >= 50 ? "#ffd60a" : "#ff3b5c"; ring.style.stroke = col; ring.style.color = col;
    $("grade").textContent = score >= 80 ? "smooth ride" : score >= 50 ? "needs repair" : "hazardous";
    const list = $("list"); list.innerHTML = "";
    if (!dets.length) list.innerHTML = '<li class="empty">No damage found above this threshold.</li>';
    dets.forEach((d, i) => {
      const li = document.createElement("li"); li.style.animationDelay = `${i * 40}ms`;
      li.innerHTML = `<span style="color:${COLORS[d.cls]}">${RoadDetect.CLASSES[d.cls]}</span><div class="bar"><i style="background:${COLORS[d.cls]};box-shadow:0 0 8px ${COLORS[d.cls]}"></i></div><span>${(d.score * 100).toFixed(0)}%</span>`;
      list.appendChild(li);
      requestAnimationFrame(() => requestAnimationFrame(() => li.querySelector("i").style.width = `${d.score * 100}%`));
    });
  }

  function countTo(el, to, animate) {
    const from = parseInt(el.textContent) || 0;
    if (!animate) { el.textContent = to; return; }
    const t0 = performance.now(), dur = 700;
    (function step(t) { const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(from + (to - from) * e); if (p < 1) requestAnimationFrame(step); })(t0);
  }
})();
