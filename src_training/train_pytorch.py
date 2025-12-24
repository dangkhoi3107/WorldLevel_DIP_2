# train_pytorch.py (Cập nhật có Augmentation)
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
DATA_DIR = "data/dataset_processed"
SEQ_LEN = 30
INPUT_DIM = 63 # Hoặc 162 nếu bạn dùng Holistic
HIDDEN_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 70
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
        # data shape: (SEQ_LEN, FEAT_DIM)
        seq_len = data.shape[0]
        mask = torch.rand(seq_len) > drop_prob
        # Nếu xui xẻo drop hết thì giữ lại
        if mask.sum() == 0: return data

        # Cách đơn giản: Giữ lại frame không bị drop và resize lại về 30
        keep_frames = data[mask]
        return Augmenter.resize_sequence(keep_frames, target_len=seq_len)

    @staticmethod
    def speed_jittering(data, speed_range=(0.8, 1.2)):
        """Thay đổi tốc độ: Co giãn chuỗi thời gian"""
        # data shape: (SEQ_LEN, FEAT_DIM)
        speed = random.uniform(*speed_range)
        # Nếu speed > 1 (nhanh hơn) -> sequence ngắn lại
        # Nếu speed < 1 (chậm hơn) -> sequence dài ra
        # Ta mô phỏng bằng cách resample
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
        # Input: (T, D) -> Cần transpose thành (1, D, T) cho interpolate
        data = data.unsqueeze(0).permute(0, 2, 1)

        # Interpolate linear
        data = F.interpolate(data, size=target_len, mode='linear', align_corners=False)

        # Transpose lại: (1, D, T) -> (T, D)
        return data.permute(0, 2, 1).squeeze(0)

def apply_augmentation(x):
    # Áp dụng ngẫu nhiên các phương pháp

    # 1. Random Temporal Crop (50% cơ hội)
    if random.random() < 0.7:
        x = Augmenter.random_temporal_crop(x, crop_size=random.randint(20, 28))

    # 2. Speed Jittering (50% cơ hội)
    if random.random() < 0.5:
        x = Augmenter.speed_jittering(x, speed_range=(0.8, 1.2))

    # 3. Frame Drop (30% cơ hội)
    if random.random() < 0.3:
        x = Augmenter.frame_drop(x, drop_prob=0.2)

    # 4. Gaussian Noise (Luôn áp dụng nhẹ hoặc 80%)
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
        x_sample = self.X[idx].clone() # Clone để không sửa dữ liệu gốc
        y_sample = self.y[idx]

        if self.augment:
            x_sample = apply_augmentation(x_sample)

        return x_sample, y_sample

# --- MODEL (Giữ nguyên GRU hoặc LSTM tùy bạn) ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- LOAD DATA ---
def load_data():
    X, y = [], []
    if not os.path.exists(DATA_DIR):
        print("Data dir not found!")
        return np.array([]), np.array([]), []

    labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    label_map = {lb: i for i, lb in enumerate(labels)}

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

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Dataset: Train có Augment, Test thì KHÔNG
    train_dataset = SignDataset(X_train, y_train, augment=True)
    test_dataset = SignDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(INPUT_DIM, HIDDEN_SIZE, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Lưu history để vẽ biểu đồ sau này
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    print("Training started...")
    for epoch in range(EPOCHS):
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

        # Validation Step
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

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")

        # Save history
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

    # Save History & Meta for Notebook
    with open("training_history.json", "w") as f:
        json.dump(history, f)

    torch.save(model.state_dict(), "./models/sign_model_augment.pth")

    # Export ONNX và Meta (giữ nguyên logic cũ của bạn)
    dummy_input = torch.randn(1, SEQ_LEN, INPUT_DIM).to(device)
    torch.onnx.export(model, dummy_input, "./models/sign_model.onnx",  # <--- SỬA TẠI ĐÂY
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

    # 3. Save Meta JSON (Sửa đường dẫn thêm ../models/)
    with open("./models/model_meta.json", "w") as f:
        json.dump({"labels": class_names}, f)

    # 4. Save History (Sửa đường dẫn thêm ../models/)
    with open("./models/training_history.json", "w") as f:
        json.dump(history, f)

    print("✅ Training complete. All files saved to ../models/")

if __name__ == "__main__":
    main()