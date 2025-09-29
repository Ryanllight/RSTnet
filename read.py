import os

# 指定文件夹路径
folder_path = "datasets/Synapse/train_npz"

# 读取文件夹中所有文件的名称（去掉后缀）
file_names = [
    os.path.splitext(file)[0]  # 去掉文件后缀
    for file in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, file))  # 确保是文件
]

# 打印结果
print("文件名列表（不含后缀）:")
for name in file_names:
    print(name)
#1720+491