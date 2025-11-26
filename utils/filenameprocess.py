import os
import re

# 设置您的日志目录路径
log_dir = r"D:\maobo\maobo_research\zhaihai\code\MyNet\BENet\result\whu\pth"

# 编译一个正则表达式，用于匹配并提取 epoch 数字
pattern = re.compile(r"BENet_BestmIoU_epoch_(\d+)_mIoU_(\d+\.\d+)?\.pth")

# 遍历文件夹中的所有文件
for filename in os.listdir(log_dir):
    # 检查文件是否符合模型文件的正则表达式
    match = pattern.match(filename)
    if match:
        # 提取 epoch 数字
        epoch = match.group(1)
        # 构建新的文件名
        new_filename = f"BENet_BestmIoU_epoch_{epoch}.pth"
        # 构建完整的原始文件路径和新文件路径
        original_path = os.path.join(log_dir, filename)
        new_path = os.path.join(log_dir, new_filename)

        # 重命名文件
        os.rename(original_path, new_path)
        print(f"Renamed '{original_path}' to '{new_path}'")

print("All files have been processed.")