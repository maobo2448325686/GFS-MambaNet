import os

# 设置图片所在的文件夹路径
folder_path = r'C:\Users\WorkStation01\Desktop\mb\data\be\WHU256\test\label'

# 设置要添加的前缀
prefix = 'test_'

# 获取文件夹中所有文件
for filename in os.listdir(folder_path):
    # 拼接完整路径
    old_path = os.path.join(folder_path, filename)

    # 只处理文件，跳过子文件夹
    if os.path.isfile(old_path):
        # 新文件名
        new_filename = prefix + filename
        new_path = os.path.join(folder_path, new_filename)

        # 重命名
        os.rename(old_path, new_path)
        # print(f'重命名: {filename} -> {new_filename}')