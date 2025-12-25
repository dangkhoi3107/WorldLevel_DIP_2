# src_training/train_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import json
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
DATA_DIR = "data/dataset_processed"
SEQ_LEN = 30
INPUT_DIM = 63
HIDDEN_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001

# --- AUGMENTATION LOGIC ---
class Augmenter:
    def __init__(self):
        pass

    @staticmethod
    def gaussian_noise(data, mean=0, std=0.01):
        """Thêm nhiễu Gaussian vào tọa độ"""
        noise = torch.randn_like(data) * std + mean
        return data + noise

    @staticmethod
    def frame_drop(data, drop_prob=0.05):
        """Ngẫu nhiên thay thế một số frame bằng frame liền trước (hoặc số 0)"""
        seq_len = data.shape[0]
        mask = torch.rand(seq_len) > drop_prob
        if mask.sum() == 0: return data

        keep_frames = data[mask]
        return Augmenter.resize_sequence(keep_frames, target_len=seq_len)

    @staticmethod
    def speed_jittering(data, speed_range=(0.8, 1.2)):
        """Thay đổi tốc độ: Co giãn chuỗi thời gian"""
        speed = random.uniform(*speed_range)
        original_len = data.shape[0]
        new_len = int(original_len / speed)
        return Augmenter.resize_sequence(data, target_len=original_len)

    @staticmethod
    def random_temporal_crop(data, crop_size=25):
        """Cắt một đoạn ngẫu nhiên trong chuỗi và resize lại về 30"""
        seq_len = data.shape[0]
        if seq_len <= crop_size: return data

        start = random.randint(0, seq_len - crop_size)
        end = start + crop_size
        crop = data[start:end]
        return Augmenter.resize_sequence(crop, target_len=seq_len)

    @staticmethod
    def resize_sequence(data, target_len=30):
        """Hàm phụ trợ để resize (interpolate) về độ dài cố định"""
        data = data.unsqueeze(0).permute(0, 2, 1)
        data = F.interpolate(data, size=target_len, mode='linear', align_corners=False)
        return data.permute(0, 2, 1).squeeze(0)

def apply_augmentation(x):
    # 1. Random Temporal Crop (50% cơ hội)
    if random.random() < 0.5:
        x = Augmenter.random_temporal_crop(x, crop_size=random.randint(20, 28))

    # 2. Speed Jittering (50% cơ hội)
    if random.random() < 0.5:
        x = Augmenter.speed_jittering(x, speed_range=(0.8, 1.2))

    # 3. Frame Drop (30% cơ hội)
    if random.random() < 0.3:
        x = Augmenter.frame_drop(x, drop_prob=0.1)

    # 4. Gaussian Noise (80% cơ hội)
    if random.random() < 0.8:
        x = Augmenter.gaussian_noise(x, std=0.005)

    return x

# --- DATASET ---
class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx].clone()
        y_sample = self.y[idx]

        if self.augment:
            x_sample = apply_augmentation(x_sample)

        return x_sample, y_sample

# --- MODEL ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True, dropout=0.3, num_layers=2)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# --- EARLY STOPPING CLASS ---
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        """
        Dừng train nếu validation loss không giảm trong 'patience' epochs.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- LOAD DATA ---
def load_data():
    X, y = [], []
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Không tìm thấy thư mục {DATA_DIR}")
        return np.array([]), np.array([]), []

    labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    label_map = {lb: i for i, lb in enumerate(labels)}
    print(f"Dataset labels: {labels}")

    for lb in labels:
        files = glob.glob(os.path.join(DATA_DIR, lb, "*.npy"))
        for f in files:
            X.append(np.load(f))
            y.append(label_map[lb])

    return np.array(X), np.array(y), labels

# --- MAIN ---
def main():
    X, y, class_names = load_data()
    if len(X) == 0: return

    print(f"Tổng số mẫu dữ liệu: {len(X)}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Dataset: Train có Augment, Test thì KHÔNG
    train_dataset = SignDataset(X_train, y_train, augment=True)
    test_dataset = SignDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GRUModel(INPUT_DIM, HIDDEN_SIZE, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # --- 1. SCHEDULER & EARLY STOPPING SETUP ---
    # Giảm LR nếu val_loss không giảm trong 5 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # Dừng hẳn nếu val_loss không giảm trong 15 epoch
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    print("🚀 Training started with Scheduler & Early Stopping...")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_loss = train_loss / len(train_loader)
        avg_acc = 100 * correct / total

        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = 100 * val_correct / val_total

        # Lấy Learning Rate hiện tại để in ra log
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}% | LR: {current_lr}")

        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        # --- 2. CẬP NHẬT SCHEDULER & CHECK EARLY STOPPING ---
        scheduler.step(avg_val_loss)

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered! Model has stopped improving.")
            break

    # --- SAVE OUTPUTS ---
    os.makedirs("./models", exist_ok=True) # Đảm bảo thư mục models tồn tại

    # 1. Save PyTorch Model
    torch.save(model.state_dict(), "./models/sign_model_augment.pth")

    # 2. Export ONNX
    dummy_input = torch.randn(1, SEQ_LEN, INPUT_DIM).to(device)
    torch.onnx.export(model, dummy_input, "./models/sign_model.onnx",
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

    # 3. Save Meta JSON
    with open("./models/model_meta.json", "w") as f:
        json.dump({"labels": class_names}, f)

    # 4. Save History
    with open("./models/training_history.json", "w") as f:
        json.dump(history, f)

    print("✅ Training complete. All files saved to ./models/")

if __name__ == "__main__":
    main()