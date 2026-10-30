// YOLO11 ONNX post-processing (shared by the browser app and the Node test).
// Copyright (c) 2026 Rishabh Aryan — MIT License
(function (root) {
  const CLASSES = ["Pothole", "Crack", "Manhole"];
  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return u > 0 ? inter / u : 0;
  }
  // out: Float32Array of shape [1, 4+nc, n]; lb: {scale, padX, padY}; returns boxes in original pixels.
  function decode(out, n, nc, lb, conf, iouThr, W, H) {
    const cand = [];
    for (let i = 0; i < n; i++) {
      let best = 0, cls = -1;
      for (let c = 0; c < nc; c++) { const s = out[(4 + c) * n + i]; if (s > best) { best = s; cls = c; } }
      if (best < conf) continue;
      const cx = out[i], cy = out[n + i], w = out[2 * n + i], h = out[3 * n + i];
      const clip = (v, m) => Math.min(Math.max(v, 0), m);
      cand.push({ cls, score: best,
        x1: clip((cx - w / 2 - lb.padX) / lb.scale, W), y1: clip((cy - h / 2 - lb.padY) / lb.scale, H),
        x2: clip((cx + w / 2 - lb.padX) / lb.scale, W), y2: clip((cy + h / 2 - lb.padY) / lb.scale, H) });
    }
    cand.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of cand) if (!keep.some(k => k.cls === d.cls && iou(k, d) > iouThr)) keep.push(d);
    return keep.slice(0, 300);
  }
  // Road health: 100 minus weighted damage share (potholes weigh most; manholes are not damage).
  function health(dets, W, H) {
    let pen = 0;
    for (const d of dets) {
      const area = ((d.x2 - d.x1) * (d.y2 - d.y1)) / (W * H);
      if (d.cls === 0) pen += 12 + 160 * area;
      else if (d.cls === 1) pen += 6 + 90 * area;
    }
    return Math.max(0, Math.round(100 - pen));
  }
  const api = { CLASSES, iou, decode, health };
  if (typeof module !== "undefined") module.exports = api; else root.RoadDetect = api;
})(this);
