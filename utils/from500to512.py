from PIL import Image
import os

input_folder  = r'C:\Users\WorkStation01\Desktop\mb\data\be\Building_Instances_of_Typical_Cities_in_China\train\label'
output_folder = r'C:\Users\WorkStation01\Desktop\mb\data\be\Building_Instances_of_Typical_Cities_in_China512\train\label'
target_size = 512

os.makedirs(output_folder, exist_ok=True)

def pad_to_512(img, size=512):
    w, h = img.size
    if w >= size and h >= size:
        return img
    # 根据模式给填充色
    if img.mode == 'L':
        fill = 0
    elif img.mode == 'RGB':
        fill = (0, 0, 0)
    elif img.mode == 'RGBA':
        fill = (0, 0, 0, 0)
    else:
        fill = 0
    new_img = Image.new(img.mode, (size, size), fill)
    new_img.paste(img, (0, 0))      # 左上角对齐
    return new_img

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.tif')):
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(output_folder, filename)
        with Image.open(src_path) as img:
            img = pad_to_512(img, target_size)
            img.save(dst_path)
print('全部补齐完成！')