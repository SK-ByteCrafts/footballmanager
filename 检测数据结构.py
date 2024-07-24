import os

def check_dataset_structure(root_dir):
    classes = ['red', 'white']
    for cls in classes:
        train_dir = os.path.join(root_dir, 'train', cls)
        val_dir = os.path.join(root_dir, 'val', cls)
        print(f"Checking {train_dir} and {val_dir}...")
        
        for dir_path in [train_dir, val_dir]:
            if not os.path.exists(dir_path):
                print(f"Directory {dir_path} does not exist.")
            else:
                files = os.listdir(dir_path)
                if len(files) == 0:
                    print(f"No files found in {dir_path}.")
                else:
                    print(f"Found {len(files)} files in {dir_path}.")
                    for file in files:
                        print(f"  {file}")

# 检查数据集目录结构
check_dataset_structure('dataset')
