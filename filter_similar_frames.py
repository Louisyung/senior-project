import numpy as np
import cv2
import glob
import os
import shutil

src_dir = 'frames'
dst_dir = 'filtered_frames'
os.makedirs(dst_dir, exist_ok=True)

frame_files = sorted(glob.glob(f'{src_dir}/frame_*.png'))
threshold = 5  # MSE 小於這個值視為相似

kept_files = [frame_files[0]]
prev_img = cv2.imread(frame_files[0])
for fname in frame_files[1:]:
    img = cv2.imread(fname)
    mse = np.mean((img.astype(np.float32) - prev_img.astype(np.float32)) ** 2)
    if mse > threshold:
        kept_files.append(fname)
        prev_img = img

# 複製保留的幀到新資料夾
for fname in kept_files:
    shutil.copy(fname, os.path.join(dst_dir, os.path.basename(fname)))

print(f"原始幀數: {len(frame_files)}, 保留幀數: {len(kept_files)}")
print(f"保留的圖片已複製到 {dst_dir}/")