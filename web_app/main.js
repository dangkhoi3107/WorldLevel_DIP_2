// =========================================================================
// SETUP DOM ELEMENTS
// =========================================================================
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("draw-canvas");
const canvasCtx = canvasEl.getContext("2d");

// UI Elements
const bigLabelEl = document.getElementById("big-prediction-label");
const confBarFillEl = document.getElementById("conf-bar-fill");
const fpsLabelEl = document.getElementById("fps-label");
const errorOverlayEl = document.getElementById("error-overlay");
const chkShowLandmarks = document.getElementById("chk-show-landmarks");

// =========================================================================
// GLOBAL STATE
// =========================================================================
const appState = {
  lastFrameTime: performance.now(),
};

// MediaPipe Hands state
let hands = null;
let handsResults = null;
let handsProcessing = false;

// ONNX Model state
let session = null;
let labelMapping = null;
let modelLoaded = false;

// =========================================================================
// CONFIGURATION
// =========================================================================
let SEQ_LEN = 30;       // Độ dài chuỗi (khớp với lúc train)
let FEAT_DIM = 63;      // 21 landmarks * 3 (x, y, z)

const PRED_STRIDE = 2;  // Dự đoán mỗi 2 frame (để giảm lag)
const VOTE_WIN = 10;    // Cửa sổ vote (Smoothing): Lấy kết quả nhiều nhất trong 10 lần gần nhất
const CONF_THRESH = 0.70; // Ngưỡng tự tin: > 70% mới hiện chữ, dưới thì làm mờ

let seqBuffer = [];     // Bộ đệm chứa 30 frame gần nhất
let frameCounter = 0;
let predHistory = [];   // Lưu lịch sử dự đoán để Vote

// Định nghĩa các đường nối khớp tay để vẽ
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],         // Ngón cái
  [0,5],[5,6],[6,7],[7,8],         // Ngón trỏ
  [0,9],[9,10],[10,11],[11,12],    // Ngón giữa
  [0,13],[13,14],[14,15],[15,16],  // Ngón áp út
  [0,17],[17,18],[18,19],[19,20],  // Ngón út
  [5,9],[9,13],[13,17],[17,5]      // Lòng bàn tay
];

// =========================================================================
// INIT CAMERA & MEDIAPIPE
// =========================================================================
async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 1280, height: 720 },
      audio: false,
    });

    videoEl.srcObject = stream;
    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });

    resizeCanvasToVideo();
    window.addEventListener("resize", resizeCanvasToVideo);

    initHands();
  } catch (err) {
    console.error(err);
    showError("Không thể truy cập Camera. Vui lòng cấp quyền.");
  }
}

function resizeCanvasToVideo() {
  if (videoEl.videoWidth && videoEl.videoHeight) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
  }
}

function showError(msg) {
  errorOverlayEl.textContent = msg;
  errorOverlayEl.hidden = false;
}

function updateFps(now) {
  const dt = now - appState.lastFrameTime;
  appState.lastFrameTime = now;
  const fps = 1000 / Math.max(dt, 1);
  fpsLabelEl.textContent = `FPS: ${fps.toFixed(0)}`;
}

// Khởi tạo MediaPipe Hands
function initHands() {
  if (typeof Hands === "undefined") {
    showError("Lỗi: Chưa load thư viện MediaPipe Hands.");
    return;
  }

  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });

  hands.onResults((results) => {
    handsResults = results;
  });

  loadGestureModel(); // Sau khi setup xong thì load model ONNX
  requestAnimationFrame(frameLoop);
}

// =========================================================================
// LOAD ONNX MODEL
// =========================================================================
async function loadGestureModel() {
  try {
    const metaPath = "./model_meta.json";
    const modelPath = "./sign_model.onnx";

    // 1. Load Metadata (Tên Class)
    const metaResp = await fetch(metaPath);
    const meta = await metaResp.json();

    if (meta.labels) {
      labelMapping = { idxToLabel: (idx) => meta.labels[idx] };
    }
    // Cập nhật config nếu trong file meta có ghi
    if (meta.seq_len) SEQ_LEN = meta.seq_len;
    if (meta.feat_dim) FEAT_DIM = meta.feat_dim;

    // 2. Load Model ONNX
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'] // Dùng WebAssembly (chạy trên CPU trình duyệt)
    });

    modelLoaded = true;
    console.log("✅ Model loaded successfully. Classes:", meta.labels);

    // Reset buffers
    seqBuffer = [];
    predHistory = [];

  } catch (err) {
    console.error(err);
    showError("Lỗi tải Model. Kiểm tra file .onnx và .json");
    modelLoaded = false;
  }
}

// =========================================================================
// FEATURE EXTRACTION & LOGIC
// =========================================================================

// Hàm trích xuất đặc trưng 63 chiều (Normalized)
function extractFeat63(landmarks) {
  const features = [];
  const wrist = landmarks[0];

  // Tính tỷ lệ dựa trên khoảng cách cổ tay -> khớp giữa (để scale invariant)
  const dx = landmarks[9].x - wrist.x;
  const dy = landmarks[9].y - wrist.y;
  const dz = (landmarks[9].z || 0) - (wrist.z || 0);
  const scale = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1.0;

  for (const lm of landmarks) {
    // Chuẩn hóa vị trí tương đối so với cổ tay và chia cho scale
    features.push((lm.x - wrist.x) / scale);
    features.push((lm.y - wrist.y) / scale);
    features.push(((lm.z || 0) - (wrist.z || 0)) / scale);
  }
  return features;
}

// Tạo frame rỗng (nếu mất tay)
function zeroFeat63() {
  return new Array(63).fill(0);
}

// Đẩy vào buffer dạng cuộn (FIFO)
function pushFrameToBuffer(feat63) {
  seqBuffer.push(feat63);
  if (seqBuffer.length > SEQ_LEN) seqBuffer.shift();
}

// Hàm vote đa số (Smoothing kết quả)
function majorityVote(arr) {
  const cnt = new Map();
  for (const x of arr) cnt.set(x, (cnt.get(x) || 0) + 1);
  let best = null, bestC = -1;
  for (const [k, v] of cnt.entries()) {
    if (v > bestC) {
      bestC = v;
      best = k;
    }
  }
  return best;
}

// --- QUAN TRỌNG: Hàm Softmax để chuyển Logits -> Xác suất % ---
function softmax(arr) {
    const maxVal = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxVal)); // Trừ max để tránh overflow
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
}

// =========================================================================
// INFERENCE LOGIC
// =========================================================================
async function runSequenceModelOnce() {
  // Chỉ chạy khi đã load model và buffer đủ 30 frame
  if (!modelLoaded || !session || seqBuffer.length < SEQ_LEN) return null;

  frameCounter++;
  if (frameCounter % PRED_STRIDE !== 0) return null; // Bỏ qua frame để giảm tải

  try {
    // 1. Flatten dữ liệu (30x63 -> mảng 1 chiều)
    const flatData = new Float32Array(SEQ_LEN * FEAT_DIM);
    for (let i = 0; i < SEQ_LEN; i++) {
      flatData.set(seqBuffer[i], i * FEAT_DIM);
    }

    // 2. Tạo Tensor Input
    // Lưu ý: Tên 'input' phải khớp với lúc export model (torch.onnx.export)
    const tensor = new ort.Tensor('float32', flatData, [1, SEQ_LEN, FEAT_DIM]);
    const feeds = { input: tensor };

    // 3. Chạy Inference
    const results = await session.run(feeds);

    // 4. Xử lý Output (Logits -> Probabilities)
    const logits = results.output.data;
    const probs = softmax(Array.from(logits)); // Chuyển sang xác suất 0.0 - 1.0

    // 5. Tìm class có xác suất cao nhất
    let bestIdx = 0, bestVal = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > bestVal) {
        bestVal = probs[i];
        bestIdx = i;
      }
    }
    return { bestIdx, conf: bestVal }; // conf giờ là số từ 0 -> 1

  } catch (e) {
    console.error("Inference Error:", e);
    return null;
  }
}

async function classifyBySequence(landmarksOrNull) {
  // 1. Cập nhật buffer dữ liệu
  if (landmarksOrNull) {
    pushFrameToBuffer(extractFeat63(landmarksOrNull));
  } else {
    pushFrameToBuffer(zeroFeat63());
  }

  // 2. Chạy model
  const pred = await runSequenceModelOnce();
  if (!pred) return null;

  // 3. Smoothing (Vote)
  predHistory.push(pred.bestIdx);
  if (predHistory.length > VOTE_WIN) predHistory.shift();
  const stableIdx = Number(majorityVote(predHistory));
  const modelLabel = labelMapping.idxToLabel(stableIdx);

  // 4. Logic hiển thị
  // pred.conf đang là xác suất của frame hiện tại (0.0 - 1.0)
  // uiConf: chuyển thành % để vẽ width
  let uiConf = Math.round(pred.conf * 100);

  if (modelLabel) {
    return { label: modelLabel, conf: pred.conf, uiConf: uiConf };
  }
  return null;
}

// =========================================================================
// DRAWING UTILS
// =========================================================================
function drawSkeleton(landmarks) {
  canvasCtx.save();
  const w = canvasEl.width;
  const h = canvasEl.height;

  // Vẽ đường nối
  canvasCtx.strokeStyle = "rgba(0, 255, 136, 0.8)";
  canvasCtx.lineWidth = 3;

  for (const [start, end] of HAND_CONNECTIONS) {
    const p1 = landmarks[start];
    const p2 = landmarks[end];
    canvasCtx.beginPath();
    canvasCtx.moveTo(p1.x * w, p1.y * h);
    canvasCtx.lineTo(p2.x * w, p2.y * h);
    canvasCtx.stroke();
  }

  // Vẽ các khớp
  canvasCtx.fillStyle = "#ff0044";
  for (const lm of landmarks) {
    canvasCtx.beginPath();
    canvasCtx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
    canvasCtx.fill();
  }
  canvasCtx.restore();
}

// =========================================================================
// MAIN LOOP
// =========================================================================
async function frameLoop(now) {
  updateFps(now);

  // Gửi frame sang MediaPipe xử lý
  if (hands && !handsProcessing) {
    handsProcessing = true;
    try {
      await hands.send({ image: videoEl });
    } finally {
      handsProcessing = false;
    }
  }

  // Xóa Canvas cũ
  canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const hasHand = handsResults && handsResults.multiHandLandmarks && handsResults.multiHandLandmarks.length > 0;
  const landmarks = hasHand ? handsResults.multiHandLandmarks[0] : null;

  // 1. Vẽ xương tay nếu bật
  if (hasHand && chkShowLandmarks.checked) {
    drawSkeleton(landmarks);
  }

  // 2. Dự đoán & Cập nhật UI
  if (modelLoaded && session) {
    const result = await classifyBySequence(landmarks);

    if (result) {
      bigLabelEl.textContent = result.label;
      confBarFillEl.style.width = `${result.uiConf}%`;

      // Logic làm mờ text nếu độ tin cậy thấp
      if (result.conf < CONF_THRESH) {
        // Nếu xác suất thấp hơn ngưỡng (70%) -> Chữ xám, thanh bar màu đỏ/cam
        bigLabelEl.style.color = "#555";
        confBarFillEl.style.backgroundColor = "#ff4444";
      } else {
        // Nếu xác suất cao -> Chữ xanh sáng, thanh bar xanh
        bigLabelEl.style.color = "#00ffcc";
        confBarFillEl.style.backgroundColor = "#00ffcc";
      }
    }
  }

  requestAnimationFrame(frameLoop);
}

// =========================================================================
// ENTRY POINT
// =========================================================================
window.addEventListener("load", () => {
  initWebcam();
});