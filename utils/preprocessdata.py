from PIL import Image
import os

input_folder  = r'C:\Users\WorkStation01\Desktop\mb\data\be\Building_Instances_of_Typical_Cities_in_China\train\label'
output_folder = r'C:\Users\WorkStation01\Desktop\mb\data\be\Building_Instances_of_Typical_Cities_in_China256\train\label'
crop_size = 256          # 目标小块尺寸
base_size = 512          # 补齐后的整张图尺寸（必须是 crop_size 的整数倍）

os.makedirs(output_folder, exist_ok=True)

def pad_to_base(img, base=512, fill=0):
    """将图像用黑色补齐到 base×base，补在右下方"""
    w, h = img.size
    if w >= base and h >= base:
        return img
    # 根据原图模式决定 fill 格式
    if img.mode == 'L':
        fill = 0                # 灰度
    elif img.mode in ('RGB', 'HSV', 'YCbCr'):
        fill = (0, 0, 0)        # 三通道
    elif img.mode == 'RGBA':
        fill = (0, 0, 0, 0)     # 带透明通道
    else:                       # 其它模式统一用 0
        fill = 0
    new_img = Image.new(img.mode, (max(w, base), max(h, base)), fill)
    new_img.paste(img, (0, 0))
    return new_img

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.tif')):
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            img = pad_to_base(img, base=base_size)   # 先补齐
            w, h = img.size                          # 此时一定是 512×512

            row_num = 1
            for i in range(0, h - crop_size + 1, crop_size):
                col_num = 1
                for j in range(0, w - crop_size + 1, crop_size):
                    box = (j, i, j + crop_size, i + crop_size)
                    crop = img.crop(box)
                    out_name = f"{os.path.splitext(filename)[0]}_{row_num}_{col_num}.png"
                    crop.save(os.path.join(output_folder, out_name))
                    col_num += 1
                row_num += 1