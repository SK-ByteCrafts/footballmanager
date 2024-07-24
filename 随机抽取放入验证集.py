import os
import shutil
import random

def get_next_filename(directory, prefix):
    existing_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    numbers = [int(f[len(prefix):-4]) for f in existing_files if f.startswith(prefix) and f[len(prefix):-4].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_number}.jpg"

def rename_files_in_directory(directory, prefix):
    files = os.listdir(directory)
    files.sort()
    for i, file in enumerate(files, start=1):
        new_filename = f"{prefix}{i}.jpg"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_filename))

def split_dataset(train_dir, val_dir, split_ratio=0.2):
    for cls in ['red', 'white']:
        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        os.makedirs(val_cls_dir, exist_ok=True)
        
        files = sorted(os.listdir(train_cls_dir), key=lambda x: int(x[len(cls):-4]))  # 确保按数字顺序排序
        num_val = int(len(files) * split_ratio)
        val_files = random.sample(files, num_val)
        
        for file in val_files:
            new_val_filename = get_next_filename(val_cls_dir, cls)
            shutil.move(os.path.join(train_cls_dir, file), os.path.join(val_cls_dir, new_val_filename))
        
        print(f"Moved {num_val} files from {train_cls_dir} to {val_cls_dir}")

def initialize_and_split():
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'

    # 确保目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for cls in ['red', 'white']:
        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

    # 重命名所有训练集的文件
    for cls in ['red', 'white']:
        rename_files_in_directory(os.path.join(train_dir, cls), cls)

    # 随机抽取训练集的一部分作为验证集，并重新命名验证集的文件
    split_dataset(train_dir, val_dir)

if __name__ == "__main__":
    initialize_and_split()
