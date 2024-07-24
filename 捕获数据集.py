import cv2
import os
import time
import numpy as np
from pinpong.board import Board, Pin

# 初始化引脚
Board().begin()
button_a = Pin(Pin.P21, Pin.IN)
button_b = Pin(Pin.P23, Pin.IN)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 确保摄像头打开成功
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建保存图像的目录
dataset_dir = 'dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

red_train_dir = os.path.join(train_dir, 'red')
white_train_dir = os.path.join(train_dir, 'white')
red_val_dir = os.path.join(val_dir, 'red')
white_val_dir = os.path.join(val_dir, 'white')

os.makedirs(red_train_dir, exist_ok=True)
os.makedirs(white_train_dir, exist_ok=True)
os.makedirs(red_val_dir, exist_ok=True)
os.makedirs(white_val_dir, exist_ok=True)

# 函数：保存图像
def save_image(image, directory, prefix):
    existing_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    file_number = len(existing_files) + 1
    filename = os.path.join(directory, f'{prefix}{file_number}.jpg')
    cv2.imwrite(filename, image)
    print(f'Saved {filename}')

# 检测颜色物体
def detect_colored_objects(frame, lower_color, upper_color, min_area=500):
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 定义颜色的HSV范围
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # 使用形态学操作去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤面积过小的轮廓
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return filtered_contours

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        break

    # 克隆一份原始图像用于保存
    original_frame = frame.copy()

    # 检测红色物体
    red_contours = detect_colored_objects(frame, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_contours += detect_colored_objects(frame, np.array([160, 100, 100]), np.array([179, 255, 255]))
    
    # 检测白色物体
    white_contours = detect_colored_objects(frame, np.array([0, 0, 200]), np.array([180, 30, 255]))

    # 绘制红色物体检测框
    for contour in red_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 绘制白色物体检测框
    for contour in white_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, "White Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 显示图像
    cv2.imshow('Frame', frame)

    # 等待按键按下
    key = cv2.waitKey(1) & 0xFF

    # 捕获红色物体图像（按下 'a' 键）
    if button_a.read_digital() == 1:
        save_image(original_frame, red_train_dir, 'red')
        time.sleep(0.5)  # 防止重复捕获

    # 捕获白色物体图像（按下 'b' 键）
    if button_b.read_digital() == 1:
        save_image(original_frame, white_train_dir, 'white')
        time.sleep(0.5)  # 防止重复捕获

    # 按下 'q' 键退出程序
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
