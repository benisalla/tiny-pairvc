// PairVC in the browser: the exported ONNX graph, run 20 ms at a time with onnxruntime-web.
// Same loop as src/pairvc/realtime.py: one chunk in, one chunk out, the model's memory
// (the state tensors) fed back between calls.

const MODEL_URL = "assets/pairvc_digits.onnx";
const SAMPLE_RATE = 16000;
const LOOKAHEAD = 640; // samples (2 latent frames of 320): the output is this late
const PREBUFFER = 2; // live mode: chunks queued before playback, against timing jitter
const CLIPS = [
  { id: "0_theo_26", word: "zero" },
  { id: "1_theo_3", word: "one" },
  { id: "2_theo_14", word: "two" },
];
const VOICES = ["voice 0 · higher (+5 semitones)", "voice 1 · deeper (−5 semitones)"];

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

class Streamer {
  static async load() {
    ort.env.wasm.numThreads = 1; // one core, as in a live app (and GitHub Pages has no cross-origin isolation)
    const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    const streamer = new Streamer(session);
    await streamer.process(new Float32Array(streamer.chunk)); // warm-up: the first run is slow
    streamer.reset();
    return streamer;
  }

  constructor(session) {
    this.session = session;
    this.meta = session.inputMetadata;
    this.chunk = this.meta.find((m) => m.name === "audio_in").shape[1];
    this.speaker = 0;
    this.pending = Promise.resolve();
    this.reset();
  }

  reset() {
    this.states = {};
    for (const m of this.meta) {
      if (m.name.startsWith("state_in")) {
        this.states[m.name] = new ort.Tensor("float32", new Float32Array(m.shape.reduce((a, b) => a * b, 1)), m.shape);
      }
    }
  }

  // One chunk; calls are queued so the model's memory is updated in order.
  process(chunk) {
    const run = this.pending.then(async () => {
      const t0 = performance.now();
      const feeds = {
        audio_in: new ort.Tensor("float32", chunk, [1, this.chunk]),
        speaker: new ort.Tensor("int64", BigInt64Array.from([BigInt(this.speaker)]), [1]),
        ...this.states,
      };
      const result = await this.session.run(feeds);
      for (const name of Object.keys(this.states)) this.states[name] = result[name.replace("state_in", "state_out")];
      return { out: result.audio_out.data, ms: performance.now() - t0 };
    });
    this.pending = run.catch(() => {});
    return run;
  }

  // A whole clip chunk by chunk, exactly as if it were live; output aligned with the input.
  async convert(audio, onProgress) {
    this.reset();
    const n = audio.length;
    const total = Math.ceil((n + LOOKAHEAD) / this.chunk) * this.chunk;
    const padded = new Float32Array(total);
    padded.set(audio);
    const out = new Float32Array(total);
    const times = [];
    for (let start = 0; start < total; start += this.chunk) {
      const { out: y, ms } = await this.process(padded.slice(start, start + this.chunk));
      out.set(y, start);
      times.push(ms);
      if (onProgress && times.length % 25 === 0) onProgress(start / total);
    }
    return { out: out.slice(LOOKAHEAD, LOOKAHEAD + n), times };
  }
}

let streamerPromise = null;
function getStreamer() {
  if (!streamerPromise) {
    streamerPromise = Streamer.load().catch((e) => {
      streamerPromise = null;
      throw e;
    });
  }
  return streamerPromise;
}

// ---------------------------------------------------------------------------
// Audio helpers: decode + resample to 16 kHz mono, WAV encoding, spectrogram
// ---------------------------------------------------------------------------

async function decodeTo16k(arrayBuffer) {
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(arrayBuffer);
  ctx.close();
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * SAMPLE_RATE), SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  return (await offline.startRendering()).getChannelData(0);
}

function wavUrl(samples, rate = SAMPLE_RATE) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((v, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 32767, true));
  return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
}

function fft(re, im) {
  // In-place radix-2 FFT (length a power of two).
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(angle * k), wi = Math.sin(angle * k);
        const a = i + k, b = a + len / 2;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

function drawSpectrogram(canvas, audio) {
  // Log-magnitude spectrogram, 0-8 kHz, 32 ms window, 8 ms hop, "magma"-like colours.
  const nfft = 512, hop = 128;
  const frames = Math.max(1, Math.floor((audio.length - nfft) / hop) + 1);
  const bins = nfft / 2;
  const window = Float32Array.from({ length: nfft }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nfft));
  const db = new Float32Array(frames * bins);
  let max = -Infinity;
  for (let f = 0; f < frames; f++) {
    const re = new Float32Array(nfft), im = new Float32Array(nfft);
    for (let i = 0; i < nfft; i++) re[i] = (audio[f * hop + i] || 0) * window[i];
    fft(re, im);
    for (let b = 0; b < bins; b++) {
      const v = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-10);
      db[f * bins + b] = v;
      if (v > max) max = v;
    }
  }
  const w = (canvas.width = frames), h = (canvas.height = bins);
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < bins; b++) {
      const t = Math.max(0, Math.min(1, (db[f * bins + b] - max + 80) / 80));
      const [r, g, bl] = magma(t);
      const p = ((h - 1 - b) * w + f) * 4;
      img.data[p] = r;
      img.data[p + 1] = g;
      img.data[p + 2] = bl;
      img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function magma(t) {
  const stops = [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]];
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  return stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f);
}

function drawTimes(canvas, times, budget) {
  // Bars: compute time of every chunk; dashed line: the real-time budget (one chunk of audio).
  const css = getComputedStyle(document.documentElement);
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const top = Math.max(budget * 1.25, ...times);
  const y = (ms) => h - 18 - (ms / top) * (h - 26);
  ctx.fillStyle = css.getPropertyValue("--accent");
  const bw = (w - 40) / times.length;
  times.forEach((ms, i) => ctx.fillRect(40 + i * bw, y(ms), Math.max(1, bw - 1), h - 18 - y(ms)));
  ctx.strokeStyle = css.getPropertyValue("--bad");
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(40, y(budget));
  ctx.lineTo(w, y(budget));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = css.getPropertyValue("--muted");
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(`${budget} ms budget`, w - 6, y(budget) - 4);
  ctx.textAlign = "left";
  ctx.fillText("0", 26, h - 18);
  ctx.fillText(`${Math.round(top)}`, 14, 12);
  ctx.fillText(`chunk → (${times.length} chunks of ${budget} ms)`, 40, h - 3);
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

// ---------------------------------------------------------------------------
// "Convert a clip": demo clip, recording or file, chunk by chunk
// ---------------------------------------------------------------------------

let lastInput = null;

async function convertAndShow(audio16k) {
  if (live) stopLive(); // one model memory: the live stream and a clip cannot share it
  lastInput = audio16k;
  const status = $("clip-status");
  status.textContent = "Loading the model (3 MB)…";
  const streamer = await getStreamer();
  streamer.speaker = Number($("voice").value);
  status.textContent = "Converting 20 ms at a time…";
  const { out, times } = await streamer.convert(audio16k, (p) => (status.textContent = `Converting… ${Math.round(100 * p)} %`));
  const chunkMs = (1000 * streamer.chunk) / SAMPLE_RATE;
  $("clip-in-audio").src = wavUrl(audio16k);
  $("clip-out-audio").src = wavUrl(out);
  $("clip-out-title").textContent = VOICES[streamer.speaker];
  drawSpectrogram($("clip-in-spec"), audio16k);
  drawSpectrogram($("clip-out-spec"), out);
  $("clip-results").hidden = false;
  drawTimes($("clip-times"), times, chunkMs);
  const m = mean(times);
  $("clip-stats").innerHTML = [
    ["audio", `${(audio16k.length / SAMPLE_RATE).toFixed(2)} s`],
    ["chunks", `${times.length} × ${chunkMs} ms`],
    ["mean compute", `${m.toFixed(2)} ms`],
    ["slowest chunk", `${Math.max(...times).toFixed(2)} ms`],
    ["real-time factor", (m / chunkMs).toFixed(3)],
  ].map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
  status.textContent = m < chunkMs
    ? `Done. Your browser converts a ${chunkMs} ms chunk in ${m.toFixed(1)} ms: fast enough for live use.`
    : `Done, but this browser needs ${m.toFixed(1)} ms per ${chunkMs} ms chunk: too slow for live use here.`;
}

async function convertDemoClip() {
  const id = $("demo-clip").value;
  const buffer = await (await fetch(`assets/audio/${id}_source.wav`)).arrayBuffer();
  await convertAndShow(await decodeTo16k(buffer));
}

let recorder = null;
async function toggleRecording() {
  const button = $("record");
  if (recorder) {
    recorder.stop();
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const parts = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => parts.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    recorder = null;
    button.textContent = "● Record";
    button.classList.remove("recording");
    const blob = new Blob(parts, { type: parts[0]?.type || "audio/webm" });
    await convertAndShow(await decodeTo16k(await blob.arrayBuffer()));
  };
  recorder.start();
  button.textContent = "■ Stop and convert";
  button.classList.add("recording");
  $("clip-status").textContent = "Recording… say a few words (try digits: the demo model was trained on them).";
}

// ---------------------------------------------------------------------------
// Live: microphone -> 20 ms chunks -> PairVC -> speakers
// ---------------------------------------------------------------------------

let live = null;

async function startLive() {
  const status = $("live-status");
  status.textContent = "Loading the model…";
  const streamer = await getStreamer();
  streamer.reset();
  streamer.speaker = Number($("voice").value);
  let ctx;
  try {
    ctx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: "interactive" });
  } catch (e) {
    status.textContent = "This browser cannot open a 16 kHz audio context. Try Chrome, Edge or Safari.";
    return;
  }
  await ctx.audioWorklet.addModule("site/io-worklet.js");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
  });
  let source;
  try {
    source = ctx.createMediaStreamSource(stream);
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
    status.textContent = "This browser cannot resample the microphone to 16 kHz (Firefox). Try Chrome, Edge or Safari.";
    return;
  }
  const node = new AudioWorkletNode(ctx, "pairvc-io", {
    processorOptions: { chunk: streamer.chunk, prebuffer: PREBUFFER },
    outputChannelCount: [2],
  });
  const times = [];
  let chunks = 0, underruns = 0;
  node.port.onmessage = async ({ data }) => {
    underruns = data.underruns;
    const { out, ms } = await streamer.process(data.chunk);
    chunks++;
    times.push(ms);
    if (times.length > 250) times.shift();
    node.port.postMessage($("bypass").checked ? data.chunk : out);
  };
  source.connect(node);
  node.connect(ctx.destination);
  const chunkMs = (1000 * streamer.chunk) / SAMPLE_RATE;
  const timer = setInterval(() => {
    if (!times.length) return;
    const device = 1000 * ((ctx.baseLatency || 0) + (ctx.outputLatency || 0));
    const latency = chunkMs + (1000 * LOOKAHEAD) / SAMPLE_RATE + PREBUFFER * chunkMs;
    $("live-stats").innerHTML = [
      ["chunks converted", chunks],
      ["mean compute", `${mean(times).toFixed(2)} ms / ${chunkMs} ms`],
      ["slowest (last 5 s)", `${Math.max(...times).toFixed(2)} ms`],
      ["real-time factor", (mean(times) / chunkMs).toFixed(3)],
      ["latency", `${Math.round(latency)} ms + ${Math.round(device)} ms audio device`],
      ["dropouts", underruns],
    ].map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
  }, 500);
  live = { ctx, stream, timer };
  $("live").textContent = "■ Stop";
  $("live").classList.add("recording");
  status.textContent = "Live: speak into your microphone (with headphones).";
}

function stopLive() {
  clearInterval(live.timer);
  live.stream.getTracks().forEach((t) => t.stop());
  live.ctx.close();
  live = null;
  $("live").textContent = "▶ Start live conversion";
  $("live").classList.remove("recording");
  $("live-status").textContent = "Stopped.";
}

// ---------------------------------------------------------------------------
// Static results: listening grid and metrics table from assets/
// ---------------------------------------------------------------------------

function buildListening() {
  const player = (id, label, cls = "") =>
    `<div class="player ${cls}"><span>${label}</span><audio controls preload="none" src="assets/audio/${id}.wav"></audio></div>`;
  $("listen-grid").innerHTML = CLIPS.map(({ id, word }) => `
    <article class="clip">
      <h3>“${word}” <small>unseen speaker</small></h3>
      ${player(`${id}_source`, "source")}
      ${[0, 1].map((k) => `
        <h4>${VOICES[k]}</h4>
        ${player(`${id}_voice${k}_untrained`, "before training", "dim")}
        ${player(`${id}_voice${k}_trained`, "after training", "main")}
        ${player(`${id}_voice${k}_teacher`, "teacher (target)")}`).join("")}
    </article>`).join("");
}

async function buildMetrics() {
  const results = await (await fetch("assets/results.json")).json();
  const rows = [
    ["timbre", "MCD (dB)", "mcd_db", "↓", (v) => v.toFixed(2)],
    ["spectrum", "mel L1", "mel_l1", "↓", (v) => v.toFixed(3)],
    ["naturalness", "DNSMOS overall (1–5)", "dnsmos_ovrl", "↑", (v) => v.toFixed(2)],
    ["", "DNSMOS speech (1–5)", "dnsmos_sig", "↑", (v) => v.toFixed(2)],
    ["pitch", "F0 RMSE (cents)", "f0_rmse_cents", "↓", (v) => v.toFixed(0)],
    ["", "voicing agreement", "voicing_agreement", "↑", (v) => `${(100 * v).toFixed(1)} %`],
  ];
  const systems = [
    ["no conversion", "no conversion"],
    ["PairVC, untrained", "before training"],
    ["PairVC, trained", "after training"],
    ["teacher", "teacher"],
  ];
  $("metrics").innerHTML = `
    <thead><tr><th></th><th>metric</th>${systems.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([group, label, key, dir, fmt]) => `
      <tr><td class="group">${group}</td><td>${label} <span class="dir">${dir}</span></td>
      ${systems.map(([name]) => `<td class="${name === "PairVC, trained" ? "main" : ""}">${fmt(results[name][key])}</td>`).join("")}</tr>`).join("")}
    </tbody>`;
}

// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  buildListening();
  buildMetrics().catch((e) => ($("metrics").textContent = `Could not load assets/results.json: ${e}`));
  $("demo-clip").innerHTML = CLIPS.map(({ id, word }) => `<option value="${id}">“${word}” (unseen speaker)</option>`).join("");
  $("voice").innerHTML = VOICES.map((v, k) => `<option value="${k}">${v}</option>`).join("");
  const report = (el) => (e) => {
    console.error(e);
    $(el).textContent = `Error: ${e.message || e}`;
  };
  $("convert-demo").onclick = () => convertDemoClip().catch(report("clip-status"));
  $("record").onclick = () => toggleRecording().catch(report("clip-status"));
  $("upload").onchange = async (e) => {
    const file = e.target.files[0];
    if (file) convertAndShow(await decodeTo16k(await file.arrayBuffer())).catch(report("clip-status"));
  };
  $("voice").onchange = () => {
    getStreamer().then((s) => (s.speaker = Number($("voice").value)));
    if (lastInput && !live) convertAndShow(lastInput).catch(report("clip-status"));
  };
  $("live").onclick = () => (live ? stopLive() : startLive().catch(report("live-status")));
});
