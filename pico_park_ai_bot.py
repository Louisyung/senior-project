import cv2
import numpy as np
import random
import time

# 載入多個角色模板圖（請確保 player1.png、player2.png 在同資料夾）
templates = [
    ('player1', cv2.imread('player1.png', cv2.IMREAD_UNCHANGED)),
    ('player2', cv2.imread('player2.png', cv2.IMREAD_UNCHANGED))
]
template_sizes = {name: tpl.shape[:2] for name, tpl in templates}

def get_game_screen_by_title(window_title):
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def find_players(frame, templates, threshold=0.8):
    results = []
    for name, template in templates:
        h, w = template.shape[:2]
        res = cv2.matchTemplate(frame, template[:, :, :3], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            results.append((name, top_left, bottom_right, max_val))
    return results

def press_random_key():
    from pynput.keyboard import Controller, Key
    keyboard = Controller()
    keys = ['right', 'up', 'space', 'd', 'a', 'w']
    key_map = {
        'right': Key.right,
        'up': Key.up,
        'space': Key.space,
        'd': 'd',
        'a': 'a',
        'w': 'w'
    }
    k = random.choice(keys)
    print(f"按下 {k}")
    keyboard.press(key_map[k])
    time.sleep(0.25)
    keyboard.release(key_map[k])

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

window_title = "PICO PARK"

print("開始亂玩 Pico Park（3 秒後）")
time.sleep(3)

while True:
    try:
        frame = get_game_screen_by_title(window_title)
    except Exception as e:
        print(e)
        break

    # 1. HSV 顏色遮罩偵測紅色區域
    mask = detect_color_blocks(frame, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # 2. 多角色模板比對
    players = find_players(frame, templates)
    for name, top_left, bottom_right, score in players:
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(frame, f"{name}:{score:.2f}", (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        print(f"{name} 位置: {top_left}, 相似度: {score:.2f}")

    press_random_key()

    cv2.imshow("Pico Park AI View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()