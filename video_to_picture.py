import cv2
import os

video_path = r'C:\Users\USER\Videos\2025-07-10 14-25-38.mkv'
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 轉成灰階並縮放到 96x96
    resized = cv2.resize(frame, (96, 96))
    cv2.imwrite(f'{output_dir}/frame_{frame_idx:05d}.png', resized)
    frame_idx += 1
cap.release()
print(f'總共擷取 {frame_idx} 幀')