# process_videos_pytorch.py
import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from data_utils import normalize_landmarks, pad_sequence

def extract_from_video(video_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    seq = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Lấy landmarks tay đầu tiên
            lms = results.multi_hand_landmarks[0]
            # Chuyển object landmark thành list [x, y, z...]
            flat_lms = []
            for lm in lms.landmark:
                flat_lms.extend([lm.x, lm.y, lm.z])

            # Chuẩn hóa ngay lập tức
            norm_feat = normalize_landmarks(flat_lms)
            seq.append(norm_feat)
        # Nếu không thấy tay, có thể bỏ qua hoặc thêm frame rỗng tùy logic

    cap.release()
    hands.close()

    if len(seq) < 5: return None # Bỏ qua nếu video quá ngắn
    return pad_sequence(seq, target_len=30)

def main(video_root, output_dir):
    # Duyệt qua thư mục: video_root/label/video.mp4
    for label in os.listdir(video_root):
        label_path = os.path.join(video_root, label)
        if not os.path.isdir(label_path): continue

        print(f"Processing label: {label}...")
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        for vid_name in os.listdir(label_path):
            vid_path = os.path.join(label_path, vid_name)
            try:
                processed_seq = extract_from_video(vid_path)
                if processed_seq is not None:
                    fname = os.path.splitext(vid_name)[0]
                    np.save(os.path.join(save_dir, f"vid_{fname}.npy"), processed_seq)
            except Exception as e:
                print(f"Error {vid_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Thư mục gốc chứa video")
    parser.add_argument("--out_dir", default="dataset_processed", help="Chung thư mục với web data")
    args = parser.parse_args()
    main(args.video_dir, args.out_dir)