
# 🖐️ AI Hand Gesture Recognition (Word-Level Detection)


This project implements a real-time Hand Gesture Recognition system capable of identifying specific sign language words directly in a web browser. The system leverages **MediaPipe** for skeletal feature extraction and a **Bi-GRU (Bidirectional Gated Recurrent Unit)** neural network for temporal sequence classification.

![Project Banner](https://via.placeholder.com/1000x300?text=AI+Hand+Gesture+Recognition+Demo)

---

## 🌟 Key Features

* **Real-time Inference:** Low-latency recognition directly via webcam feed.
* **Two-Stage Architecture:**
    * *Stage 1:* Binary Sign Detector (detects if a hand is signing vs. idle).
    * *Stage 2:* Word-Level Classifier (classifies the specific word from 27 vocabulary classes).
* **Robust AI Model:** Trained with advanced **Data Augmentation** techniques (Gaussian Noise, Speed Jittering, Random Temporal Crop) to handle camera instability and speed variations.
* **Web-Based Deployment:** powered by **ONNX Runtime Web (WASM)**, requiring no backend server interaction for inference.
* **Custom Data Collection:** Integrated web tool for recording and labeling custom datasets.

## 📂 Project Structure

```text
ROOT/
├── data/                      # Raw videos and processed Numpy arrays
├── models/                    # Trained models (.pth, .onnx) and training history
├── src_training/              # Python source code for training
│   ├── train_pytorch.py       # Main training script (with Augmentation)
│   ├── evaluate.ipynb         # Evaluation notebook (Charts & Confusion Matrix)
│   └── process_videos...py    # Data preprocessing script
├── tools/
│   └── data_collector/        # HTML/JS tool for collecting dataset via webcam
├── web_app/                   # Main Frontend Application
│   ├── index.html
│   ├── main.js                # Inference logic using ONNX Runtime
│   └── style.css
└── requirements.txt           # Python dependencies

```

## 🚀 Installation & Setup

### 1. Environment Preparation

Ensure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd WorldLevel_DIP_2

# Create a virtual environment (Recommended)
conda create -n WorldLevel_DIP_2 python=3.10
conda activate WorldLevel_DIP_2

# Install dependencies
pip install -r requirements.txt

```

### 2. Data Collection (Optional)

If you want to train on your own gestures:

1. Open `tools/data_collector/data-collector.html` in your browser.
2. Enter a label (e.g., "Hello") and press **Record**.
3. Save the downloaded files into `data/dataset_processed/[Label]/`.

### 3. Training Pipeline

**Step 1: Data Preprocessing**
Convert raw landmarks/videos into standardized sequences (fixed 30 frames):

```bash
python  src_training/process_videos_pytorch.py --video_dir  "data/videos" --out_dir "data/dataset_processed"

```

**Step 2: Train the Model**
Run the training script (includes on-the-fly augmentation):

```bash
python src_training/train_pytorch.py

```

*Output:* The script will generate `sign_model.onnx` and `sign_model_augment.pth` in the `models/` directory.

**Step 3: Evaluation**
Open `src_training/evaluate.ipynb` in Jupyter Notebook to visualize Learning Curves, Confusion Matrices, and Per-Class Accuracy.

## 🌐 Web Application Deployment

To run the recognition system locally:

1. **Update Model:** Copy the generated `sign_model.onnx` and `model_meta.json` from the `models/` folder and paste them into the `web_app/` folder.
2. **Launch Server:**
* **Using VS Code:** Right-click `web_app/index.html` and select **"Open with Live Server"**.




## 📊 Performance Metrics

Evaluated on the WLASL subset (Test Set):

| Metric | Score | Description |
| --- | --- | --- |
| **Accuracy** | **81.0%** | Overall classification accuracy |
| **Precision** | **84.0%** | Positive predictive value |
| **F1-Score** | **80.0%** | Harmonic mean of Precision and Recall |

*Detailed analysis and charts are available in the final project report.*

## 🛠️ Tech Stack

* **Core:** Python, PyTorch
* **Computer Vision:** MediaPipe Hands
* **Deployment:** ONNX Runtime Web
* **Techniques:** Bi-GRU, Stratified Sampling, Data Augmentation.

---
