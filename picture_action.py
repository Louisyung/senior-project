import cv2
import numpy as np
import glob

frame_files = sorted(glob.glob('filtered_frames/frame_*.png'))
actions = []

for fname in frame_files:
    img = cv2.imread(fname)
    cv2.imshow('frame', img)
    cv2.waitKey(100)  # 讓視窗有機會刷新
    print("請輸入此幀的動作編號（player1, player2），例如 0 1：")
    s = input("player1 player2: ")  # 例如輸入 0 1
    try:
        a1, a2 = map(int, s.strip().split())
        actions.append([a1, a2])
    except:
        print("格式錯誤，請重新輸入")
        continue
cv2.destroyAllWindows()
np.save('actions.npy', np.array(actions))