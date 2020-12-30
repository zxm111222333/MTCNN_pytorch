# MTCNN人脸检测
import os
from PIL import Image, ImageDraw
from nets import *
from tools import utils
import torch
import numpy
from torchvision import transforms
import time

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 全局变量
# P网络:
PYRAMID_SCALE_COEFFICIENT = 0.71  # P网络图片金字塔缩放系数
PNET_CONF_THRESHOLD = 0.9  # P网络置信度阈值
PNET_NMS_THRESHOLD = 0.5  # P网络非极大值抑制阈值
# R网络:
RNET_CONF_THRESHOLD = 0.9  # R网络置信度阈值
RNET_NMS_THRESHOLD = 0.5  # R网络非极大值抑制阈值
# O网络:
ONET_CONF_THRESHOLD = 0.9999  # O网络置信度阈值
ONET_NMS_THRESHOLD = 0.5  # O网络非极大值抑制阈值


# 自定义检测器
class Detector:
    # 初始化
    def __init__(self, pnet_param, rnet_param, onet_param, is_cuda=True):
        super(Detector, self).__init__()
        self.is_cuda = is_cuda
        # 初始化模型
        # # simple or normal
        # self.pnet = PNet()
        # self.rnet = RNet()
        # self.onet = ONet()

        # other
        self.pnet = Pnet2()
        self.rnet = Rnet2()
        self.onet = Onet2()

        # # bn
        # self.pnet = PNet3()
        # self.rnet = RNet3()
        # self.onet = ONet3()
        # # Set model to evaluate mode
        # self.pnet.eval()
        # self.rnet.eval()
        # self.onet.eval()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))
        if is_cuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        # 图片数据类型转换
        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # 检测图片
    # 1.将image放到P网络检测，得到经过nms后的候选框pnet_boxes
    # 2.将pnet_boxes放到R网络检测，得到经过nms后的候选框rnet_boxes
    # 3.将rnet_boxes放到R网络检测，得到经过nms后的候选框onet_boxes
    def __call__(self, image):
        pnet_start = time.time()
        # 1.将image放到P网络检测，得到经过nms后的候选框pnet_boxes
        pnet_boxes = self.__pnet_detect(image)
        # 若pnet_boxes为[]，直接返回
        if pnet_boxes.shape[0] == 0:
            return numpy.array([])

        rnet_start = time.time()
        # 2.将pnet_boxes放到R网络检测，得到经过nms后的候选框rnet_boxes
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        # 若rnet_boxes为[]，直接返回
        if rnet_boxes.shape[0] == 0:
            return numpy.array([])

        onet_start = time.time()
        # 3.将rnet_boxes放到R网络检测，得到经过nms后的候选框onet_boxes
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        # 若onet_boxes为[]，直接返回
        if onet_boxes.shape[0] == 0:
            return numpy.array([])
        onet_end = time.time()

        # 打印网络检测时间
        print("total:", onet_end - pnet_start, "pnet:", rnet_start - pnet_start, "rnet:", onet_start - rnet_start,
              "onet:", onet_end - onet_start)

        # return pnet_boxes
        # return rnet_boxes
        return onet_boxes

    # P网络检测函数，返回pnet_boxes
    # 1.创建候选框列表
    # 2.图片进行金字塔缩放，缩放图片大小满足min(w, h)>=12
    # 3.将缩放图片输入P网络得到输出: 置信度+边框偏移量+关键点偏移量
    # 4.先通过置信度阈值过滤一部分；剩余部分通过"特征图反算"得到候选框，并添加到候选框列表中
    # 5.最后，通过非极大值抑制排除部分的候选框，并返回新的候选框列表
    def __pnet_detect(self, image):
        # 1.创建候选框列表
        boxes = []

        # 2.图片进行金字塔缩放，缩放图片大小满足min(w, h)>=12
        img = image
        img_w, img_h = img.size
        min_side_len = min(img_w, img_h)
        scale = 1  # 初始缩放系数为1
        while min_side_len >= 12:
            # 3.将缩放图片输入P网络得到输出: 置信度+边框偏移量+关键点偏移量
            # 将图片转换为张量tensor
            img_data = self.__image_transform(img)
            # 将一张图片的(C,H,W)转换为(N,C,H,W)
            img_data.unsqueeze_(dim=0)  # 对自身升维操作
            # 放到模型中进行训练
            if self.is_cuda:
                img_data = img_data.cuda()
            cls_output, bbox_offset_output, landmark_offset_output = self.pnet(img_data)

            # 4.先通过置信度阈值过滤一部分；剩余部分通过"特征图反算"得到候选框，并加入到候选框列表boxes中
            # cls_output: (1, 1, h, w)，bbox_offset_output: (1, 4, h, w)，landmark_offset_output: (1, 10, h, w)
            cls = cls_output[0][0].cpu().detach()
            bbox_offset = bbox_offset_output[0].cpu().detach()
            # 根据置信度阈值筛选
            index = torch.where(cls > PNET_CONF_THRESHOLD)
            idxs = torch.stack((index[0], index[1]), dim=1)

            for idx in idxs:
                # 通过"特征图反算"得到候选框，并添加到候选框列表中
                boxes.append(self.__back_cal_box(idx, cls[idx[0], idx[1]], bbox_offset[:, idx[0], idx[1]], scale))

            # 图片缩放，控制循环条件
            scale *= PYRAMID_SCALE_COEFFICIENT
            _w, _h = int(img_w * scale), int(img_h * scale)
            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        # 5.最后，通过非极大值抑制排除部分的候选框，并返回新的候选框列表
        new_boxes = utils.nms(numpy.array(boxes), PNET_NMS_THRESHOLD)
        # print(len(boxes), len(new_boxes))
        return new_boxes

    # 特征图反算，返回候选框
    # PNet的多层级联步长stride=2，多层级联核大小kernel_size=12
    def __back_cal_box(self, idx, cls, bbox_offset, scale, stride=2, kernel_size=12):
        # 根据特征图索引(x, y)，反算感受野区域: 左上角点(x*s, y*s)、右下角点(x*s+k, y*s+k)
        x, y = idx[1], idx[0]
        _x1 = x * stride
        _y1 = y * stride
        _x2 = x * stride + kernel_size
        _y2 = y * stride + kernel_size
        # 再根据边框偏移量，计算所预测的候选框
        _w = _h = kernel_size
        x1 = _x1 + _w * bbox_offset[0]
        y1 = _y1 + _h * bbox_offset[1]
        x2 = _x2 + _w * bbox_offset[2]
        y2 = _y2 + _h * bbox_offset[3]
        # 考虑到图片缩放，计算在原图中的候选框位置
        return [x1 / scale, y1 / scale, x2 / scale, y2 / scale, cls]

    # R网络检测函数，返回rnet_boxes
    # 1.创建候选框列表
    # 2.根据pnet_boxes抠出所有的图片，resize成24*24，并转换为(N,C,H,W)格式
    # 3.将转换后的图片数据传入R网络得到输出: 置信度+边框偏移量+关键点偏移量
    # 4.先通过置信度阈值过滤一部分；剩余部分计算对应的候选框，并添加到候选框列表中
    # 5.最后，通过非极大值抑制排除部分的候选框，并返回新的候选框列表
    def __rnet_detect(self, image, pnet_boxes):
        # 1.创建候选框列表
        boxes = []

        # 2.根据pnet_boxes抠出所有的图片，resize成24*24，并转换为(N,C,H,W)格式
        pnet_boxes = utils.convert_to_square(pnet_boxes)  # 转换为正方形

        img_dataset = []  # 存放抠图列表
        for box in pnet_boxes:
            crop_img = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            resize_img = crop_img.resize((24, 24))
            # 转换为(N,C,H,W)格式
            img_data = self.__image_transform(resize_img)
            # print(img_data.shape)
            img_dataset.append(img_data)
        # print(len(img_dataset))
        # 组装为矩阵
        img_dataset = torch.stack(img_dataset)
        # print(img_dataset.shape)

        # 3.将转换后的图片数据传入R网络得到输出: 置信度+边框偏移量+关键点偏移量
        if self.is_cuda:
            img_dataset = img_dataset.cuda()
        cls_output, bbox_offset_output, landmark_offset_output = self.rnet(img_dataset)

        # 4.先通过置信度阈值过滤一部分；剩余部分计算对应的候选框，并添加到候选框列表中
        _cls, _bbox_offset = cls_output.cpu().detach().numpy(), bbox_offset_output.cpu().detach().numpy()
        select_mask = _cls[:, 0] > RNET_CONF_THRESHOLD
        select_cls = _cls[select_mask]
        select_bbox_offset = _bbox_offset[select_mask]
        select_pnet_boxes = pnet_boxes[select_mask]
        for cls, bbox_offset, pnet_box in zip(select_cls, select_bbox_offset, select_pnet_boxes):
            _x1, _y1, _x2, _y2 = pnet_box[0], pnet_box[1], pnet_box[2], pnet_box[3]
            _w, _h = _x2 - _x1, _y2 - _y1
            # 图片resize不改变偏移量
            x1 = _x1 + _w * bbox_offset[0]
            y1 = _y1 + _h * bbox_offset[1]
            x2 = _x2 + _w * bbox_offset[2]
            y2 = _y2 + _h * bbox_offset[3]

            boxes.append((x1, y1, x2, y2, cls))

        # 5.最后，通过非极大值抑制排除部分的候选框，并返回新的候选框列表
        new_boxes = utils.nms(numpy.array(boxes), RNET_NMS_THRESHOLD)
        # print(len(boxes), len(new_boxes))
        return new_boxes

    # O网络检测函数，返回onet_boxes
    # 1.创建候选框列表
    # 2.根据rnet_boxes抠出所有的图片，resize成48*48，并转换为(N,C,H,W)格式
    # 3.将转换后的图片数据传入O网络得到输出: 置信度+边框偏移量+关键点偏移量
    # 4.先通过置信度阈值过滤一部分；剩余部分计算对应的候选框，并添加到候选框列表中
    # 5.最后，通过非极大值抑制排除部分的候选框，并返回新的候选框列表
    def __onet_detect(self, image, rnet_boxes):
        # 1.创建候选框列表
        boxes = []

        # 2.根据rnet_boxes抠出所有的图片，resize成48*48，并转换为(N,C,H,W)格式
        rnet_boxes = utils.convert_to_square(rnet_boxes)  # 转换为正方形
        img_dataset = []  # 存放抠图列表
        for box in rnet_boxes:
            crop_img = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            resize_img = crop_img.resize((48, 48))
            # 转换为(N,C,H,W)格式
            img_data = self.__image_transform(resize_img)
            img_dataset.append(img_data)
        # 转换为矩阵
        img_dataset = torch.stack(img_dataset)

        # 3.将转换后的图片数据传入O网络得到输出: 置信度+边框偏移量+关键点偏移量
        if self.is_cuda:
            img_dataset = img_dataset.cuda()
        cls_output, bbox_offset_output, landmark_offset_output = self.onet(img_dataset)

        # 4.先通过置信度阈值过滤一部分；剩余部分计算对应的候选框，并添加到候选框列表中
        _cls, _bbox_offset = cls_output.cpu().detach().numpy(), bbox_offset_output.cpu().detach().numpy()
        _landmark_offset_output = landmark_offset_output.cpu().detach().numpy()
        select_mask = _cls[:, 0] > ONET_CONF_THRESHOLD
        select_cls = _cls[select_mask]
        select_bbox_offset = _bbox_offset[select_mask]
        select_landmark_offset = _landmark_offset_output[select_mask]
        select_rnet_boxes = rnet_boxes[select_mask]
        for cls, bbox_offset, landmark_offset, rnet_box in zip(select_cls, select_bbox_offset, select_landmark_offset,
                                                               select_rnet_boxes):
            _x1, _y1, _x2, _y2 = rnet_box[0], rnet_box[1], rnet_box[2], rnet_box[3]
            _w, _h = _x2 - _x1, _y2 - _y1
            # 图片resize不改变偏移量
            x1 = _x1 + _w * bbox_offset[0]
            y1 = _y1 + _h * bbox_offset[1]
            x2 = _x2 + _w * bbox_offset[2]
            y2 = _y2 + _h * bbox_offset[3]

            lefteye_x = _x1 + _w * landmark_offset[0]
            lefteye_y = _y1 + _h * landmark_offset[1]
            righteye_x = _x1 + _w * landmark_offset[2]
            righteye_y = _y1 + _h * landmark_offset[3]
            nose_x = _x1 + _w * landmark_offset[4]
            nose_y = _y1 + _h * landmark_offset[5]
            leftmouth_x = _x1 + _w * landmark_offset[6]
            leftmouth_y = _y1 + _h * landmark_offset[7]
            rightmouth_x = _x1 + _w * landmark_offset[8]
            rightmouth_y = _y1 + _h * landmark_offset[9]

            boxes.append((
                x1, y1, x2, y2, cls, lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x,
                leftmouth_y, rightmouth_x, rightmouth_y))

        # 5.最后，通过非极大值抑制排除部分的候选框，并返回新的候选框列表
        new_boxes = utils.nms(numpy.array(boxes), ONET_NMS_THRESHOLD, isMin=True)
        # print(len(boxes), len(new_boxes))
        return new_boxes


if __name__ == '__main__':
    test_image_path = r"D:\MTCNN\data\MTCNN-Pytorch\detect_img"

    # # simple------
    # detector = Detector(r"D:\MTCNN\MTCNN-Pytorch\result\simple\checkpoint\pnet.pt",
    #                     r"D:\MTCNN\MTCNN-Pytorch\result\simple\checkpoint\rnet.pt",
    #                     r"D:\MTCNN\MTCNN-Pytorch\result\simple\checkpoint\onet.pt")

    # # normal------
    # detector = Detector(r"D:\MTCNN\MTCNN-Pytorch\result\normal\checkpoint\pnet.pt",
    #                     r"D:\MTCNN\MTCNN-Pytorch\result\normal\checkpoint\rnet.pt",
    #                     r"D:\MTCNN\MTCNN-Pytorch\result\normal\checkpoint\onet.pt")

    # other------效果最好
    detector = Detector(r"D:\MTCNN\MTCNN-Pytorch\result\other\pnet.t",
                        r"D:\MTCNN\MTCNN-Pytorch\result\other\rnet.t",
                        r"D:\MTCNN\MTCNN-Pytorch\result\other\onet.t")

    # # bn------
    # detector = Detector(r"D:\MTCNN\MTCNN-Pytorch\result\bn\checkpoint\p_35.pt",
    #                     r"D:\MTCNN\MTCNN-Pytorch\result\bn\checkpoint\r_66.pt",
    #                     r"D:\MTCNN\MTCNN-Pytorch\result\bn\checkpoint\o_46.pt")

    for filename in os.listdir(test_image_path):
        img_path = os.path.join(test_image_path, filename)

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        onet_boxes = detector(img)
        for box in onet_boxes:
            x1, y1, x2, y2, cls = box[0], box[1], box[2], box[3], box[4]
            lefteye_x, lefteye_y = box[5], box[6]
            righteye_x, righteye_y = box[7], box[8]
            nose_x, nose_y = box[9], box[10]
            leftmouth_x, leftmouth_y = box[11], box[12]
            rightmouth_x, rightmouth_y = box[13], box[14]

            # print("bbox:", (x1, y1, x2, y2), "conf:", cls)
            # if cls == 1:
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
            plt.plot(lefteye_x, lefteye_y, ".")
            plt.plot(righteye_x, righteye_y, ".")
            plt.plot(nose_x, nose_y, ".")
            plt.plot(leftmouth_x, leftmouth_y, ".")
            plt.plot(rightmouth_x, rightmouth_y, ".")
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        plt.imshow(img)
        plt.show()
