import os
import shutil

def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def initialize_directories():
    directories = [
        'dataset/train/red',
        'dataset/train/white',
        'dataset/val/red',
        'dataset/val/white',
        '输出结果/',
        'models/'
    ]
    
    # 清空目录
    for directory in directories:
        print(f"Clearing directory: {directory}")
        clear_directory(directory)
        os.makedirs(directory, exist_ok=True)
        
    print("所有数据均已清除，进入格式化状态")

if __name__ == "__main__":
    initialize_directories()
