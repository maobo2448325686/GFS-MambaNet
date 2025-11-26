import re
import matplotlib.pyplot as plt

# 读取日志文件
with open('../result_sparse_4/guiyang/trainval.txt', 'r', encoding='utf-8') as f:
    log_text = f.read()

# 提取数据
epochs = []
train_losses = []
val_losses = []

# 匹配 Train 和 Val 的 loss
train_matches = re.findall(r'Train Epoch #(\d+).*total_loss=([\d.]+)', log_text)
val_matches = re.findall(r'Val Epoch #(\d+).*total_loss_val=([\d.]+)', log_text)

# 转换为字典方便对齐
train_dict = {int(e): float(l) for e, l in train_matches}
val_dict = {int(e): float(l) for e, l in val_matches}

# 对齐epoch
all_epochs = sorted(set(train_dict.keys()) & set(val_dict.keys()))
for e in all_epochs:
    epochs.append(e)
    train_losses.append(train_dict[e])
    val_losses.append(val_dict[e])

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
plt.plot(epochs, val_losses, label='Validation Loss', marker='x', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()