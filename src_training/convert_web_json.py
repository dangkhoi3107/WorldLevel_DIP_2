# convert_web_json.py
import json
import numpy as np
import os
import argparse
from data_utils import normalize_landmarks, pad_sequence

def process_web_data(json_path, output_dir="dataset_processed"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data.get("samples", [])
    print(f"📂 Đọc {len(samples)} mẫu từ {json_path}")

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for item in samples:
        label = item['label']
        raw_seq = item['seq'] # Đây là dữ liệu thô từ JS

        # Chuẩn hóa từng frame
        processed_seq = []
        for frame in raw_seq:
            # Frame từ JS là list 63 phần tử phẳng
            norm_feat = normalize_landmarks(frame)
            processed_seq.append(norm_feat)

        # Pad về độ dài 30
        final_seq = pad_sequence(processed_seq, target_len=30)

        # Lưu file .npy
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = item.get('timestamp', count)
        np.save(os.path.join(save_dir, f"web_{timestamp}.npy"), final_seq)
        count += 1

    print(f"✅ Đã convert xong {count} mẫu vào thư mục '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="File JSON xuất từ data-collector.html")
    args = parser.parse_args()
    process_web_data(args.json)