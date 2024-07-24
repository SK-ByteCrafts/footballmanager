import os
import math

def read_accelerations_from_file(filename):
    accelerations = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            if "ax" in lines[i+1] and "ay" in lines[i+2]:
                ax = float(lines[i+1].strip().split(': ')[1].split()[0])
                ay = float(lines[i+2].strip().split(': ')[1].split()[0])
                accelerations.append((ax, ay))
    return accelerations

def calculate_total_acceleration(accelerations):
    total_accelerations = []
    for ax, ay in accelerations:
        a = math.sqrt(ax**2 + ay**2)
        total_accelerations.append(a)
    return total_accelerations

def save_total_acceleration_to_file(filename, total_accelerations):
    with open(filename, 'w') as file:
        for i, a in enumerate(total_accelerations):
            file.write(f'最终加速度 for object {i+1}: {a:.2f} pixels/s^2\n')
    print(f'Saved total acceleration to {filename}')

# 文件路径
red_filename = '/root/输出结果/red分量加速度.txt'
white_filename = '/root/输出结果/white分量加速度.txt'
red_output_filename = '/root/输出结果/red最终计算加速度.txt'
white_output_filename = '/root/输出结果/white最终计算加速度.txt'

# 读取加速度数据
red_accelerations = read_accelerations_from_file(red_filename)
white_accelerations = read_accelerations_from_file(white_filename)

# 计算总加速度
red_total_accelerations = calculate_total_acceleration(red_accelerations)
white_total_accelerations = calculate_total_acceleration(white_accelerations)

# 保存总加速度到文件
save_total_acceleration_to_file(red_output_filename, red_total_accelerations)
save_total_acceleration_to_file(white_output_filename, white_total_accelerations)
