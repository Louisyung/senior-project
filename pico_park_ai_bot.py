import cv2
import numpy as np
import random
import time

def get_game_screen_by_title(window_title):
    """
    擷取指定視窗標題的畫面（自動定位視窗，不需手動填座標）
    需先安裝: pip install pygetwindow mss pillow
    """
    import pygetwindow as gw
    import mss
    from PIL import Image

    win = None
    for w in gw.getAllWindows():
        if window_title in w.title:
            win = w
            break
    if win is None:
        raise Exception("找不到視窗！請確認視窗標題。")
    bbox = (win.left, win.top, win.right, win.bottom)
    with mss.mss() as sct:
        sct_img = sct.grab(bbox)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame

def detect_color_blocks(img, lower, upper):
    # 使用顏色範圍來偵測像素區域
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def press_random_key():
    from pynput.keyboard import Controller, Key
    keyboard = Controller()
    keys = [ 'right', 'up', 'space', 'd']
    key_map = {
        'right': Key.right,
        'up': Key.up,
        'space': Key.space,
         'd': 'd'
    }
    k = random.choice(keys)
    print(f"按下 {k}")
    keyboard.press(key_map[k])
    time.sleep(0.25)
    keyboard.release(key_map[k])

# 設定顏色範圍（ex. 紅色角色）
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

window_title = "PICO PARK"  # 請改成你的遊戲視窗標題

print("開始亂玩 Pico Park（3 秒後）")
time.sleep(3)

while True:
    try:
        frame = get_game_screen_by_title(window_title)
    except Exception as e:
        print(e)
        break

    mask = detect_color_blocks(frame, lower_red, upper_red)

    # 偵測到紅色區域（角色），畫框
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:  # 篩掉雜訊
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    press_random_key()

    # 顯示畫面
    cv2.imshow("Pico Park AI View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()