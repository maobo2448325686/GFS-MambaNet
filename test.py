import torchvision.transforms as Transforms
import torch
import glob
import os

from ptflops import get_model_complexity_info
from torch import nn

from model.BENet import BENet
from utils.evaluation import Evaluation
from utils.evaluation import Index
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def img(pr, gt, filename):
    pr = (pr > 0.5).float()
    pr = pr[0, 0].cpu().detach().numpy()  # 转换为 NumPy 数组

    # 确保 gt 的值是二值的（0 或 1）
    gt = (gt > 0.5).astype(np.uint8)

    tp = np.logical_and(pr == 1, gt == 1)
    fp = np.logical_and(pr == 1, gt == 0)
    tn = np.logical_and(pr == 0, gt == 0)
    fn = np.logical_and(pr == 0, gt == 1)

    image = np.zeros([512, 512, 3], dtype=np.uint8)

    image[tp] = [255, 255, 255]  # 真正例 (白色)
    image[fp] = [232, 68, 69]      # 假正例 (红色)
    image[tn] = [0, 0, 0]        # 真负例 (黑色)
    image[fn] = [0, 173, 239]      # 假负例 (绿色)

    res_image = Image.fromarray(image)
    save_path = os.path.join(image_pre_dir, filename)
    # print()
    res_image.save(save_path)


#
def test_net(net, device, test_path, ModelName='BENet', epochs=100):
    for epoch in range(epochs):
        # test
        if epoch >= 0:
            test(net, device, epoch, test_path)


def test(net, device, modelpath, test_DataPath):
    # 加载特定 epoch 的模型参数
    model_path = modelpath
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model file not found: {model_path}")

    net.eval()


    # 计算参数量和flops
    macs, params = get_model_complexity_info(
        net,
        input_res=(3, 512, 512),  # 两个 3 通道图像拼接在一起
        as_strings=False,
        print_per_layer_stat=False
    )
    # 转换为 G 和 M
    flops_G = macs * 0.5 / 1e9
    params_M = params / 1e6
    print(f'Params: {params_M:.2f} M, FLOPs: {flops_G:.2f} G')

    # 匹配 image 文件夹中的 .png 和 .tif 文件
    image_png = glob.glob(os.path.join(test_DataPath, 'image/*.png'))
    image_tif = glob.glob(os.path.join(test_DataPath, 'image/*.tif'))
    image = image_png + image_tif  # 合并两个列表

    # 匹配 label 文件夹中的 .png 和 .tif 文件
    label_png = glob.glob(os.path.join(test_DataPath, 'label/*.png'))
    label_tif = glob.glob(os.path.join(test_DataPath, 'label/*.tif'))
    label = label_png + label_tif  # 合并两个列表



    trans = Transforms.Compose([Transforms.ToTensor()])
    IoU, c_IoU, uc_IoU, OA, Precision, Recall, F1 = 0., 0., 0., 0., 0., 0., 0.
    num = 0
    TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or = 0., 0., 0., 0., 0., 0.
    test_acc = open(log_dir + '/test.txt', 'a')

    with tqdm(total=len(image), desc='test', colour='blue') as t:
        for test_path, label_path in zip(image, label):
            num += 1
            filename = os.path.basename(test_path)
            test_img = cv2.imread(test_path)
            test_label_old = cv2.imread(label_path)
            test_label = cv2.cvtColor(test_label_old, cv2.COLOR_BGR2GRAY)
            test_img = trans(test_img)

            test_img = test_img.unsqueeze(0)

            test_img = test_img.to(device=device, dtype=torch.float32)

            pred = net(test_img)

            # image_pre
            img(pred, test_label, filename)

            # acquire result-vmamba
            pred = np.array(pred.data.cpu()[0])[0]
            # binary map
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            monfusion_matrix = Evaluation(label=test_label, pred=pred)
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
                'OA': '%.4f' % OA,
                'mIoU': '%.4f' % IoU,
                'c_IoU': '%.4f' % c_IoU,
                'uc_IoU': '%.4f' % uc_IoU,
                'PRE': '%.4f' % Precision,
                'REC': '%.4f' % Recall,
                'F1': '%.4f' % F1})
            t.update(1)
    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    test_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                   str(float('%2f' % (c_IoU))) + ',' +
                   'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                   'PRE = ' + str(float('%2f' % (Precision))) + ',' +
                   'REC = ' + str(float('%2f' % (Recall))) + ',' +
                   'F1 = ' + str(float('%2f' % (F1))) + '\n')
    test_acc.close()
    return OA, IoU


if __name__ == '__main__':

    net = BENet()
    net.to(device=device)

    log_dir = r"result/whu"
    image_pre_dir = r"result/whu/prediction"

    os.makedirs(image_pre_dir, exist_ok=True)


    test_path = r"C:\Users\WorkStation01\Desktop\mb\data\be\WHU512old\test"
    modelpath = r"result/whu/best_f1_96.05996871483723.pth"
    test(net, device, modelpath=modelpath, test_DataPath=test_path)

