const videoEl = document.getElementById("webcam");
const overlayCanvasEl = document.getElementById("overlay-canvas");
const overlayCtx = overlayCanvasEl.getContext("2d");

const statusTextEl = document.getElementById("status-text");
const currentLabelEl = document.getElementById("current-label");
const currentLabelNameEl = document.getElementById("current-label-name");

const errorOverlayEl = document.getElementById("error-overlay");

// NEW UI
const labelSelectEl = document.getElementById("label-select");
const classCountsEl = document.getElementById("class-counts");
const countTotalEl = document.getElementById("count-total");

// =========================
// 27 classes
// =========================
const CLASS_NAMES = [
  "apple",
  "bad",
  "because",
  "bird",
  "black",
  "candy",
  "cousin",
  "deaf",
  "delicious",
  "drink",
  "government",
  "hot",
  "like",
  "mother",
  "no",
  "orange",
  "pizza",
  "silly",
  "sweet",
  "tell",
  "theory",
  "white",
  "who",
  "why",
  "woman",
  "yes",
  "yesterday",
];

// =========================
// Sequence collection config
// =========================
const SEQ_LEN = 30;         // fixed T per clip
const TARGET_FPS = 15;      // sample fps
const FEATURE_DIM = 63;     // 21*(x,y,z)
const MIN_NONZERO_FRAMES = 5;

// Data collection state: each item is ONE CLIP (sequence)
const collectedData = []; // {label, seq:[T,63], raw_len, detected_frames, timestamp}
let isRecording = true;

// Current selection
let currentClassIndex = 0;

// Current clip state
let isClipRecording = false;
let clipLabel = null;
let clipFrames = [];      // raw frames: [[63], ...]
let lastSampleTime = 0;

// MediaPipe Hands state
let hands = null;
let handsResults = null;
let handsProcessing = false;

// =========================
// Init UI
// =========================
function initLabelSelect() {
  // populate dropdown
  labelSelectEl.innerHTML = "";
  for (let i = 0; i < CLASS_NAMES.length; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${i + 1}. ${CLASS_NAMES[i]}`;
    labelSelectEl.appendChild(opt);
  }
  labelSelectEl.value = String(currentClassIndex);

  labelSelectEl.addEventListener("change", () => {
    setCurrentClassIndex(parseInt(labelSelectEl.value, 10));
  });

  rebuildCountsUI();
  updateStatus(`Ready. Choose class with [ ] or dropdown. Hold R to record a clip.`);
}

function setCurrentClassIndex(i) {
  if (Number.isNaN(i)) return;
  currentClassIndex = Math.max(0, Math.min(CLASS_NAMES.length - 1, i));
  labelSelectEl.value = String(currentClassIndex);

  // If currently recording, do not auto switch label mid-clip
  if (!isClipRecording) {
    updateStatus(`Current class: "${getCurrentLabel()}". Hold R to record.`);
  }
}

function getCurrentLabel() {
  return CLASS_NAMES[currentClassIndex];
}

function rebuildCountsUI() {
  classCountsEl.innerHTML = "";
  for (const label of CLASS_NAMES) {
    const row = document.createElement("div");
    row.className = "stat";

    const left = document.createElement("span");
    left.className = "label";
    left.textContent = `${label}:`;

    const right = document.createElement("span");
    right.className = "count";
    right.id = `count-${label}`;
    right.textContent = "0";

    row.appendChild(left);
    row.appendChild(right);
    classCountsEl.appendChild(row);
  }
}

// =========================
// Webcam + canvas
// =========================
async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    videoEl.srcObject = stream;

    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resizeCanvas();
        window.addEventListener("resize", resizeCanvas);
        resolve();
      };
    });

    initHands();
  } catch (err) {
    console.error("Failed to initialize webcam:", err);
    showError("Could not access the webcam. Please allow camera permissions and refresh the page.");
  }
}

function resizeCanvas() {
  const container = videoEl.parentElement;
  overlayCanvasEl.width = container.clientWidth;
  overlayCanvasEl.height = container.clientHeight;
}

function getDisplayedVideoRect() {
  const container = videoEl.parentElement;
  const containerAspect = container.clientWidth / container.clientHeight;
  const videoAspect = videoEl.videoWidth / videoEl.videoHeight;

  let displayWidth, displayHeight, offsetX, offsetY;

  if (containerAspect > videoAspect) {
    displayHeight = container.clientHeight;
    displayWidth = displayHeight * videoAspect;
    offsetX = (container.clientWidth - displayWidth) / 2;
    offsetY = 0;
  } else {
    displayWidth = container.clientWidth;
    displayHeight = displayWidth / videoAspect;
    offsetX = 0;
    offsetY = (container.clientHeight - displayHeight) / 2;
  }

  return { displayWidth, displayHeight, offsetX, offsetY };
}

// =========================
// MediaPipe Hands
// =========================
function initHands() {
  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  hands.onResults(onHandsResults);
  frameLoop();
}

function onHandsResults(results) {
  handsResults = results;
  handsProcessing = false;

  drawLandmarks(results);

  // sequence sampling here
  if (!isRecording || !isClipRecording) return;

  const now = performance.now();
  const minInterval = 1000 / TARGET_FPS;
  if (lastSampleTime && now - lastSampleTime < minInterval) return;
  lastSampleTime = now;

  const hasHand = results?.multiHandLandmarks?.length > 0;

  if (hasHand) {
    const landmarks = results.multiHandLandmarks[0];
    clipFrames.push(featFromLandmarks63(landmarks));
  } else {
    clipFrames.push(zeros63());
  }
}

function drawLandmarks(results) {
  overlayCtx.clearRect(0, 0, overlayCanvasEl.width, overlayCanvasEl.height);

  const hasHand = results?.multiHandLandmarks?.length > 0;
  if (!hasHand) return;

  const landmarks = results.multiHandLandmarks[0];
  const videoRect = getDisplayedVideoRect();

  overlayCtx.strokeStyle = "#00ff88";
  overlayCtx.lineWidth = 2;

  HAND_CONNECTIONS.forEach(([start, end]) => {
    const s = landmarks[start];
    const e = landmarks[end];
    overlayCtx.beginPath();
    overlayCtx.moveTo(
      videoRect.offsetX + s.x * videoRect.displayWidth,
      videoRect.offsetY + s.y * videoRect.displayHeight
    );
    overlayCtx.lineTo(
      videoRect.offsetX + e.x * videoRect.displayWidth,
      videoRect.offsetY + e.y * videoRect.displayHeight
    );
    overlayCtx.stroke();
  });

  for (const lm of landmarks) {
    overlayCtx.fillStyle = "#00ff88";
    overlayCtx.beginPath();
    overlayCtx.arc(
      videoRect.offsetX + lm.x * videoRect.displayWidth,
      videoRect.offsetY + lm.y * videoRect.displayHeight,
      3,
      0,
      2 * Math.PI
    );
    overlayCtx.fill();
  }
}

// =========================
// Feature helpers
// =========================
function featFromLandmarks63(landmarks) {
  const features = new Array(FEATURE_DIM);
  let j = 0;
  for (const lm of landmarks) {
    features[j++] = lm.x;
    features[j++] = lm.y;
    features[j++] = lm.z;
  }
  return features;
}

function zeros63() {
  return new Array(FEATURE_DIM).fill(0);
}

function padOrResample(frames) {
  if (!frames || frames.length === 0) {
    return Array.from({ length: SEQ_LEN }, () => zeros63());
  }
  if (frames.length === SEQ_LEN) return frames;

  if (frames.length < SEQ_LEN) {
    const out = frames.slice();
    while (out.length < SEQ_LEN) out.push(zeros63());
    return out;
  }

  const out = [];
  for (let i = 0; i < SEQ_LEN; i++) {
    const idx = Math.floor((i * (frames.length - 1)) / (SEQ_LEN - 1));
    out.push(frames[idx]);
  }
  return out;
}

function countNonZeroFrames(frames) {
  let c = 0;
  for (const f of frames) {
    let s = 0;
    for (let i = 0; i < f.length; i++) s += Math.abs(f[i]);
    if (s > 1e-6) c++;
  }
  return c;
}

// =========================
// Clip controls
// =========================
function startClip(label) {
  if (!isRecording) return;
  if (isClipRecording) return;

  isClipRecording = true;
  clipLabel = label;
  clipFrames = [];
  lastSampleTime = 0;

  currentLabelNameEl.textContent = label;
  currentLabelEl.classList.remove("hidden");
  updateStatus(`Recording clip for "${label}"... (release R to stop)`);
}

function stopClip({ save = true } = {}) {
  if (!isClipRecording) return;

  const label = clipLabel;
  const rawLen = clipFrames.length;

  isClipRecording = false;
  clipLabel = null;
  currentLabelEl.classList.add("hidden");

  if (!save) {
    clipFrames = [];
    updateStatus(`Clip discarded.`);
    return;
  }

  const nonZero = countNonZeroFrames(clipFrames);
  if (nonZero < MIN_NONZERO_FRAMES) {
    clipFrames = [];
    updateStatus(`Clip not saved: too few detected frames (${nonZero}).`);
    return;
  }

  const seq = padOrResample(clipFrames);
  collectedData.push({
    label,
    seq,
    raw_len: rawLen,
    detected_frames: nonZero,
    timestamp: Date.now(),
  });

  clipFrames = [];
  updateStats();
  updateStatus(`Saved "${label}" clip (#${getCountForLabel(label)}). Next class: use ]`);
}

// =========================
// Stats
// =========================
function updateStats() {
  const counts = {};
  for (const lb of CLASS_NAMES) counts[lb] = 0;

  for (const s of collectedData) {
    if (counts[s.label] != null) counts[s.label]++;
  }

  // total
  countTotalEl.textContent = String(collectedData.length);

  // per class
  for (const lb of CLASS_NAMES) {
    const el = document.getElementById(`count-${lb}`);
    if (el) el.textContent = String(counts[lb]);
  }
}

function getCountForLabel(label) {
  return collectedData.filter((s) => s.label === label).length;
}

// =========================
// Keyboard controls
// =========================
document.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    isRecording = !isRecording;

    if (!isRecording) {
      if (isClipRecording) stopClip({ save: true });
      updateStatus("Paused");
    } else {
      updateStatus(`Recording enabled. Current class: "${getCurrentLabel()}". Hold R to record.`);
    }
    return;
  }

  // export / clear
  if (e.key === "e" || e.key === "E") {
    exportData();
    return;
  }
  if (e.key === "c" || e.key === "C") {
    if (confirm("Clear all collected data?")) clearData();
    return;
  }

  // navigation: [ and ]
  if (e.key === "[") {
    e.preventDefault();
    if (e.repeat) return;
    setCurrentClassIndex(currentClassIndex - 1);
    return;
  }
  if (e.key === "]") {
    e.preventDefault();
    if (e.repeat) return;
    setCurrentClassIndex(currentClassIndex + 1);
    return;
  }

  // record: hold R
  if (e.key === "r" || e.key === "R") {
    e.preventDefault();
    if (e.repeat) return;
    startClip(getCurrentLabel());
  }
});

document.addEventListener("keyup", (e) => {
  if (e.key === "r" || e.key === "R") {
    e.preventDefault();
    stopClip({ save: true });
  }
});

// =========================
// Export / Clear
// =========================
function exportData() {
  if (collectedData.length === 0) {
    alert("No data to export! Record some clips first.");
    return;
  }

  const label_to_id = {};
  CLASS_NAMES.forEach((lb, i) => (label_to_id[lb] = i));

  const exportObj = {
    seq_len: SEQ_LEN,
    target_fps: TARGET_FPS,
    feature_dim: FEATURE_DIM,
    labels: CLASS_NAMES,
    label_to_id,
    samples: collectedData,
  };

  const dataStr = JSON.stringify(exportObj, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `asl-seq-data-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(url);

  updateStatus(`Exported ${collectedData.length} clips.`);
}

function clearData() {
  stopClip({ save: false });
  collectedData.length = 0;
  updateStats();
  updateStatus(`Data cleared. Current class: "${getCurrentLabel()}". Hold R to record.`);
}

function updateStatus(message) {
  statusTextEl.textContent = message;
}

function showError(message) {
  errorOverlayEl.textContent = "";
  const p = document.createElement("p");
  p.textContent = message;
  errorOverlayEl.appendChild(p);
  errorOverlayEl.hidden = false;
}

// =========================
// Main frame loop
// =========================
function frameLoop() {
  if (!handsProcessing && videoEl.readyState === videoEl.HAVE_ENOUGH_DATA) {
    handsProcessing = true;
    hands.send({ image: videoEl });
  }
  if (handsResults) drawLandmarks(handsResults);
  requestAnimationFrame(frameLoop);
}

// MediaPipe Hand Connections
const HAND_CONNECTIONS = [
  [0, 1],[1, 2],[2, 3],[3, 4],
  [0, 5],[5, 6],[6, 7],[7, 8],
  [0, 9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5, 9],[9,13],[13,17],[17, 5],
];

// Initialize on load
window.addEventListener("load", () => {
  initLabelSelect();
  initWebcam();
});
