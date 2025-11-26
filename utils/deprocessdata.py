# -*- coding: utf-8 -*-
"""
通用图像拼接脚本
支持两种命名：
  1) 行列式：xxx_行_列.扩展名
  2) 顺序式：xxx_序号.扩展名（从左到右、从上到下连续编号）
"""
import os
import re
import cv2
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from math import ceil, sqrt

# ===== 只需改这里两个路径 =====
SRC_ROOT = r'C:\Users\WorkStation01\Desktop\mb\data\be\WHU256\val\label'      # 小图目录
DST_ROOT = r'C:\Users\WorkStation01\Desktop\mb\data\be\WHU512old\val\label'      # 大图保存目录
# ============================

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
EXT_STR = '|'.join(map(re.escape, IMG_EXT))

# 预编译两套正则
# RE_ROW_COL = re.compile(r'^(?P<prefix>.+)_(\d+)_(\d+)(' + EXT_STR + r')$', re.I)   # 行列式
RE_INDEX  = re.compile(r'^(?P<prefix>.+)_(\d+)(' + EXT_STR + r')$', re.I)          # 顺序式

def parse_name(fname):
    """
    返回 (prefix, row, col, full_path) 或 (prefix, index, None, full_path)
    如果都匹配失败则返回 None
    """
    # m = RE_ROW_COL.match(fname)
    # if m:
    #     prefix, r, c = m.group('prefix'), int(m.group(2)), int(m.group(3))
    #     return prefix, r, c, True               # True 表示是行列式
    m = RE_INDEX.match(fname)
    if m:
        prefix, idx = m.group('prefix'), int(m.group(2))
        return prefix, idx, None, False         # False 表示是顺序式
    return None

# def scan_groups(root):
#     """
#     返回 dict: prefix -> {'type': 'rowcol'/'index', 'tiles': [(r,c,path)] 或 [(idx,path)]}
#     """
#     groups = {}
#     for dirpath, _, files in os.walk(root):
#         for f in files:
#             res = parse_name(f)
#             if not res:
#                 continue
#             prefix, r, c, is_rowcol = res
#             fp = os.path.join(dirpath, f)
#             if is_rowcol:
#                 groups.setdefault(prefix, {'type': 'rowcol', 'tiles': []})['tiles'].append((r, c, fp))
#             else:
#                 groups.setdefault(prefix, {'type': 'index', 'tiles': []})['tiles'].append((r, fp))
#     return groups



def scan_groups(root):
    """
    返回 dict: prefix -> {'type': 'rowcol'/'index', 'tiles': [(r,c,path)] 或 [(idx,path)]}
    """
    groups = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            res = parse_name(f)
            if not res:
                continue
            prefix, r, c, is_rowcol = res
            fp = os.path.join(dirpath, f)
            groups.setdefault(prefix, {'type': 'index', 'tiles': []})['tiles'].append((r, fp))
    return groups

def compute_rows_cols(index_list, start_zero=True):
    """
    根据顺序号列表自动推断“正方形”行列数
    例：0~15 -> 4×4；1~12 -> 3×4
    """
    if start_zero:
        n = max(index_list) + 1
    else:
        n = max(index_list)
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)
    return rows, cols

def merge_one(job):
    prefix, info = job
    tiles = info['tiles']
    # ---------- 1. 统一成 (row,col,path) 形式 ----------
    if info['type'] == 'rowcol':
        # 已有 row/col
        items = tiles
    else:
        # 顺序式：先推断行列
        idx_list = [t[0] for t in tiles]
        start_zero = min(idx_list) == 0
        rows, cols = compute_rows_cols(idx_list, start_zero)
        # 建立 (row,col) 映射
        items = []
        for idx, fp in tiles:
            if start_zero:
                idx0 = idx
            else:
                idx0 = idx - 1
            r = idx0 // cols + 1
            c = idx0 % cols + 1
            items.append((r, c, fp))

    # ---------- 2. 拼接 ----------
    rows = max(r for r, _, _ in items)
    cols = max(c for _, c, _ in items)
    h, w = cv2.imread(items[0][2]).shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for r, c, fp in items:
        tile = cv2.imread(fp)
        canvas[(r-1)*h:r*h, (c-1)*w:c*w] = tile

    # ---------- 3. 保存 ----------
    os.makedirs(DST_ROOT, exist_ok=True)
    ext = os.path.splitext(items[0][2])[1]
    out_path = os.path.join(DST_ROOT, prefix + ext)
    cv2.imwrite(out_path, canvas)
    return out_path

def main():
    groups = scan_groups(SRC_ROOT)
    if not groups:
        print('未找到任何匹配命名规则的图片')
        return
    # 多进程 + 进度条
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(merge_one, groups.items()),
                            total=len(groups), desc='拼接进度'))
    print(f'全部完成！共 {len(results)} 张大图已保存到 {DST_ROOT}')

if __name__ == '__main__':
    main()