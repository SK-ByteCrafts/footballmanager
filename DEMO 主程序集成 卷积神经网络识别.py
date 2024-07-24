import sys
import time
import numpy as np
import cv2
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pinpong.board import Board, Pin
from pinpong.extension.unihiker import *

# 定义卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: red, white

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load('models/best_simple_cnn.pth'))
model.eval()

# 定义数据转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 打开摄像头
cap = cv2.VideoCapture(0)

# 初始化引脚
Board().begin()
button_a = Pin(Pin.P21, Pin.IN)

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 确保摄像头打开成功
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

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

# 初始化追踪变量
prev_time = None
prev_positions = {'red': [], 'white': []}
velocities = {'red': [], 'white': []}
accelerations = {'red': [], 'white': []}
latest_acceleration = {'red': [], 'white': []}

def calculate_average_acceleration(acceleration_deque):
    avg_ax = np.mean([a[0] for a in acceleration_deque])
    avg_ay = np.mean([a[1] for a in acceleration_deque])
    return avg_ax, avg_ay

def save_acceleration_to_file(color, accelerations_list):
    filename = f'/root/输出结果/{color}分量加速度.txt'
    with open(filename, 'w') as file:
        for i, acc in enumerate(accelerations_list):
            file.write(f'该物体平均加速度/a dx/dy for {color} object {i+1}:\n')
            file.write(f'加速度分量ax: {acc[0]:.2f} pixels/s^2\n')
            file.write(f'加速度分量ay: {acc[1]:.2f} pixels/s^2\n')
    print(f'Saved average acceleration to {filename}')

# 减少采样频率的计数器
frame_counter = 0
sampling_interval = 5  # 每5帧采样一次

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        break

    frame_counter += 1

    # 当前时间
    current_time = time.time()

    # 转换为RGB图像并进行分类
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb_frame)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        label = predicted.item()

    if label == 0:
        color = 'red'
        contours = detect_colored_objects(frame, np.array([0, 100, 100]), np.array([10, 255, 255]))
        contours += detect_colored_objects(frame, np.array([160, 100, 100]), np.array([179, 255, 255]))
    else:
        color = 'white'
        contours = detect_colored_objects(frame, np.array([0, 0, 200]), np.array([180, 30, 255]))

    # 筛选出同时满足颜色和CNN模型条件的物体
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        roi = frame[y:y+h, x:x+w]
        roi_tensor = transform(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            outputs = model(roi_tensor)
            _, predicted = torch.max(outputs, 1)
            if (predicted.item() == 0 and color == 'red') or (predicted.item() == 1 and color == 'white'):
                filtered_contours.append(contour)

    # 绘制检测框并计算速度和加速度
    latest_acceleration[color] = []
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w / 2, y + h / 2)
        if color == 'red':
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"Red Object {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, f"White Object {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if len(prev_positions[color]) > i:
            if prev_positions[color][i] is not None and frame_counter % sampling_interval == 0:
                dt = current_time - prev_time
                dx = center[0] - prev_positions[color][i][0]
                dy = center[1] - prev_positions[color][i][1]
                velocity = (dx / dt, dy / dt)
                if len(velocities[color]) <= i:
                    velocities[color].append(deque(maxlen=10))
                velocities[color][i].append(velocity)
                
                if len(velocities[color][i]) > 1:
                    dvx = velocities[color][i][-1][0] - velocities[color][i][-2][0]
                    dvy = velocities[color][i][-1][1] - velocities[color][i][-2][1]
                    acceleration = (dvx / dt, dvy / dt)
                    if len(accelerations[color]) <= i:
                        accelerations[color].append(deque(maxlen=10))
                    accelerations[color][i].append(acceleration)
                    avg_acceleration = calculate_average_acceleration(accelerations[color][i])
                    latest_acceleration[color].append(avg_acceleration)
            prev_positions[color][i] = center
        else:
            prev_positions[color].append(center)
            velocities[color].append(deque(maxlen=10))
            accelerations[color].append(deque(maxlen=10))

    prev_time = current_time

    # 显示最新的加速度值
    for i, acc in enumerate(latest_acceleration['red']):
        cv2.putText(frame, f"Red Acc {i+1}: ({acc[0]:.2f}, {acc[1]:.2f})", (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for i, acc in enumerate(latest_acceleration['white']):
        cv2.putText(frame, f"White Acc {i+1}: ({acc[0]:.2f}, {acc[1]:.2f})", (10, 50 + len(latest_acceleration['red']) * 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 显示图像
    cv2.imshow('Frame', frame)

    # 等待按键按下
    key = cv2.waitKey(1) & 0xFF

    if button_a.read_digital() == 1:
        # 计算并保存红色物体的平均加速度
        if accelerations['red']:
            avg_acc_red = [calculate_average_acceleration(acc) for acc in accelerations['red']]
            save_acceleration_to_file('red', avg_acc_red)
        
        # 计算并保存白色物体的平均加速度
        if accelerations['white']:
            avg_acc_white = [calculate_average_acceleration(acc) for acc in accelerations['white']]
            save_acceleration_to_file('white', avg_acc_white)

    # 按下 'q' 键退出程序
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
