import os
import matplotlib.pyplot as plt
import numpy as np

def read_acceleration_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    ax_values = []
    ay_values = []
    
    for line in lines:
        if line.startswith('加速度分量ax:'):
            ax = float(line.split(':')[1].strip().split()[0])
            ax_values.append(ax)
        elif line.startswith('加速度分量ay:'):
            ay = float(line.split(':')[1].strip().split()[0])
            ay_values.append(ay)
    
    return ax_values[:2], ay_values[:2]  # 只使用前两个数据

def plot_acceleration_data(ax_values, ay_values, title):
    x = np.arange(len(ax_values))
    width = 0.35  # 条形宽度

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, ax_values, width, label='ax')
    bars2 = ax.bar(x + width/2, ay_values, width, label='ay')

    # 添加标签、标题和自定义 x 轴刻度标签
    ax.set_xlabel('Sample')
    ax.set_ylabel('Acceleration (pixels/s^2)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()

    # 在每个条形图上方显示高度值
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移3个点
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    fig.tight_layout()
    plt.show()

def process_acceleration_files(directory):
    files_processed = False
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")  # 调试输出
            ax_values, ay_values = read_acceleration_data(file_path)
            print(f"ax_values: {ax_values}, ay_values: {ay_values}")  # 调试输出
            title = os.path.splitext(filename)[0]
            plot_acceleration_data(ax_values, ay_values, title)
            files_processed = True
    
    if not files_processed:
        print("No txt files found in the specified directory.")

if __name__ == "__main__":
    # 替换为包含加速度数据txt文件的目录
    output_dir = '/root/输出结果'
    process_acceleration_files(output_dir)
