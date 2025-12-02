from utils.dataset_new import ISBI_Loader, mixup_collate_fn
from torch import optim, nn
import torchvision.transforms as Transforms
import torch.utils.data as data
import time
import torch
import glob
import os
from model.BENet import BENet
from utils.evaluation import Evaluation
from utils.evaluation import Index
from utils.loss import dice_loss
import cv2
import numpy as np
from tqdm import tqdm



alpha = 1
#
def train_net(net, device, data_path, val_path, test_path, log_dir, ModelName='Net-vmamba', epochs=100, batch_size=4,
              lr=0.0001, resume_from=None):
    # print(net)
    # Load dataset
    isbi_dataset = ISBI_Loader(data_path, transform=Transforms.ToTensor())
    train_loader = data.DataLoader(dataset=isbi_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=mixup_collate_fn(alpha=0.4))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
                                                     gamma=0.9)

    BCE_loss = nn.BCELoss()

    f_loss = open(log_dir + '/train_loss.txt', 'w')
    f_time = open(log_dir + '/train_time.txt', 'w')

    start_epoch = 0
    if resume_from:
        # 加载预训练模型
        if os.path.exists(resume_from):
            print(f"Loading model from {resume_from}")
            net.load_state_dict(torch.load(resume_from, map_location=device))
            # 提取批次号
            start_epoch = 56
            print("Switching to SGD optimizer")
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)  # 使用当前的学习率
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], gamma=0.9)
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Pretrained model file {resume_from} not found. Starting from scratch.")


    best_f1 = 0.

    for epoch in range(start_epoch + 1, epochs + 1):

        # 切换到 SGD 优化器
        # if epoch == 66:
        #     print("Switching to SGD optimizer")
        #     optimizer = optim.SGD(net.parameters(), lr=lr * (0.9 ** 10), momentum=0.9)  # 使用当前的学习率
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 95, 110, 125, 140], gamma=0.9)


        net.train()
        total_loss = 0
        num = int(0)

        starttime = time.time()

        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch), colour='white') as t:
            for image, label in train_loader:
                optimizer.zero_grad()

                image = image.to(device=device)
                label = label.to(device=device)

                out = net(image)

                # compute loss
                dice_Loss = dice_loss(out, label)
                bce_loss = BCE_loss(out, label)
                loss = alpha * dice_Loss + (1 - alpha) * bce_loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                num += 1
                t.set_postfix({'lr': '%.8f' % optimizer.param_groups[0]['lr'],
                               'total_loss': '%.4f' % (total_loss / num), })
                if num == len(train_loader):
                    f_loss.write('epoch: ' + str(epoch) + '    total_loss: ' + str(total_loss / num) + '\n')
                t.update(1)
        # learning rate delay
        scheduler.step()

        endtime = time.time()
        if epoch == 0:
            f_time.write('each epoch time\n')
        f_time.write(str(epoch) + ',' + str(starttime) + ',' + str(endtime) + ',' + str(
            float('%4f' % (endtime - starttime))) + '\n')
        # val
        if epoch >= 0:
            with torch.no_grad():
                _, _, F1 = val(net, device, epoch, val_path)
                if F1 > best_f1:
                    best_f1 = F1
                    modelpath_best = 'best_' + 'f1_' + str(best_f1) + '.pth'
                    torch.save(net.state_dict(), log_dir + "/" + modelpath_best)
                    print("Best model saved! epoch: " + str(epoch))
        modelpath_last = 'last_model' + '.pth'
        torch.save(net.state_dict(), log_dir + "/" + modelpath_last)

    f_loss.close()
    f_time.close()


def val(net, device, epoc, val_DataPath):
    net.eval()
    # 匹配 image 文件夹中的 .png 和 .tif 文件
    image_png = glob.glob(os.path.join(val_DataPath, 'image/*.png'))
    image_tif = glob.glob(os.path.join(val_DataPath, 'image/*.tif'))
    image = image_png + image_tif  # 合并两个列表

    # 匹配 label 文件夹中的 .png 和 .tif 文件
    label_png = glob.glob(os.path.join(val_DataPath, 'label/*.png'))
    label_tif = glob.glob(os.path.join(val_DataPath, 'label/*.tif'))
    label = label_png + label_tif  # 合并两个列表

    trans = Transforms.Compose([Transforms.ToTensor()])
    IoU, c_IoU, uc_IoU, OA, Precision, Recall, F1 = 0, 0, 0, 0, 0, 0, 0
    num = 0
    total_loss_val = 0.
    BCE_loss = nn.BCELoss()
    TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or = 0, 0, 0, 0, 0, 0
    val_acc = open(log_dir + '/val_acc.txt', 'a')
    val_acc.write('===============================' + 'epoch=' + str(epoc) + '==============================\n')
    with tqdm(total=len(image), desc='Val Epoch #{}'.format(epoc), colour='yellow') as t:
        for val_path, label_path in zip(image, label):
            num += 1

            val_img = cv2.imread(val_path)
            val_label = cv2.imread(label_path)
            val_label = cv2.cvtColor(val_label, cv2.COLOR_BGR2GRAY)
            val_img = trans(val_img)

            val_img = val_img.unsqueeze(0)
            val_img = val_img.to(device=device, dtype=torch.float32)

            mask = trans(val_label).to(device=device, dtype=torch.float32)

            pred = net(val_img)

            dice_loss_val = dice_loss(pred, mask)
            bce_loss_val = BCE_loss(pred, mask.unsqueeze(1))
            loss_val = alpha * dice_loss_val+ (1 - alpha) * bce_loss_val

            total_loss_val += loss_val.item()

            # acquire result-vmamba
            pred = np.array(pred.data.cpu()[0])[0]
            # binary map
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            monfusion_matrix = Evaluation(label=val_label, pred=pred)
            TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
            TPSum += TP
            TNSum += TN
            FPSum += FP
            FNSum += FN
            C_Sum_or += c_num_or
            UC_Sum_or += uc_num_or

            if num > 0:
                Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
                IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
                OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()

            t.set_postfix({
                'total_loss_val': '%.4f' % (total_loss_val / num),
                'OA': '%.2f' % OA,
                'mIoU': '%.2f' % IoU,
                'c_IoU': '%.2f' % c_IoU,
                'uc_IoU': '%.2f' % uc_IoU,
                'PRE': '%.2f' % Precision,
                'REC': '%.2f' % Recall,
                'F1': '%.2f' % F1})
            t.update(1)

    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    val_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                  str(float('%2f' % (c_IoU))) + ',' +
                  'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                  'PRE = ' + str(float('%2f' % (Precision))) + ',' +
                  'REC = ' + str(float('%2f' % (Recall))) + ',' +
                  'F1 = ' + str(float('%2f' % (F1))) + '\n')
    val_acc.close()
    return OA, IoU, F1


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = BENet()
    net.to(device=device)
    log_dir = "result_sparse_4/chinacity"

    os.makedirs(log_dir, exist_ok=True)

    train_path = r"C:\Users\WorkStation01\Desktop\mb\data\be\Building_Instances_of_Typical_Cities_in_China512\train"
    val_path = r"C:\Users\WorkStation01\Desktop\mb\data\be\Building_Instances_of_Typical_Cities_in_China512\val"
    test_path = r"C:\Users\WorkStation01\Desktop\mb\data\be\Building_Instances_of_Typical_Cities_in_China512\test"

    # 指定从哪个预训练模型继续训练
    resume_from = None
    train_net(net, device, train_path, val_path, test_path, log_dir, batch_size=8, epochs=100, lr=0.0001, resume_from=resume_from)
