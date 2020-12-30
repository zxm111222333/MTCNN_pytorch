# MTCNN人脸检测
"""
待优化：cv2的resize比PIL.Iamge的resize快。
"""
import time
import numpy
import torch
from torchvision import transforms
from optimize_nets import PNet2, RNet2, ONet2
from tools import utils
# 绘图
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import os

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
        self.pnet = PNet2()
        self.rnet = RNet2()
        self.onet = ONet2()
        # 加载模型参数
        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))
        # 将模型放到GPU上
        if is_cuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        # 图片数据类型转换
        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

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



        return onet_boxes

    def __pnet_detect(self, image):

        boxes = []

        img = numpy.array(image)
        img_h, img_w, _ = img.shape
        min_side_len = min(img_w, img_h)
        scale = 1  # 初始缩放系数为1
        while min_side_len >= 12:

            img_data = self.__image_transform(img)
            img_data.unsqueeze_(dim=0)  
            if self.is_cuda:
                img_data = img_data.cuda()

            cls_output, bbox_offset_output = self.pnet(img_data)

            cls = cls_output[0][0].cpu().detach().numpy()
            bbox_offset = bbox_offset_output[0].cpu().detach().numpy()

            index = numpy.where(cls > PNET_CONF_THRESHOLD)
            idxs = numpy.stack((index[0], index[1]), axis=1)
            if idxs.shape[0] > 0:
                boxes.append(
                    self.__back_cal_boxs(idxs, cls[idxs[:, 0], idxs[:, 1]], bbox_offset[:, idxs[:, 0], idxs[:, 1]],

            scale *= PYRAMID_SCALE_COEFFICIENT
            _w, _h = int(img_w * scale), int(img_h * scale)
       
            img = cv2.resize(img, (_w, _h))
            min_side_len = min(_w, _h)
   
        if len(boxes) > 0:
            boxes = numpy.concatenate(boxes, axis=1).T
        else:
            boxes = numpy.array(boxes)

        return utils.nms(boxes, PNET_NMS_THRESHOLD)

    def __back_cal_boxs(self, idxs, cls, bbox_offset, scale, stride=2, kernel_size=12):
   
        x, y = idxs[:, 1], idxs[:, 0]  # (n)
        _x1 = x * stride
        _y1 = y * stride
        _x2 = x * stride + kernel_size
        _y2 = y * stride + kernel_size
        _w = _h = kernel_size
        x1 = _x1 + _w * bbox_offset[0]
        y1 = _y1 + _h * bbox_offset[1]
        x2 = _x2 + _w * bbox_offset[2]
        y2 = _y2 + _h * bbox_offset[3]  # (n)
  
        return numpy.array([x1 / scale, y1 / scale, x2 / scale, y2 / scale, cls])
    def __rnet_detect(self, image, pnet_boxes):

        pnet_boxes = utils.convert_to_square(pnet_boxes)  

        img_dataset = []  # 存放抠图列表
        for box in pnet_boxes:
            crop_img = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            resize_img = crop_img.resize((24, 24))
            # 转换为(N,C,H,W)格式
            img_data = self.__image_transform(resize_img)
            img_dataset.append(img_data)
        # 组装为矩阵
        img_dataset = torch.stack(img_dataset)

        if self.is_cuda:
            img_dataset = img_dataset.cuda()
        cls_output, bbox_offset_output = self.rnet(img_dataset)
        _cls, _bbox_offset = cls_output.cpu().detach().numpy(), bbox_offset_output.cpu().detach().numpy()
        select_mask = _cls[:, 0] > RNET_CONF_THRESHOLD
        select_cls = _cls[select_mask]
        select_bbox_offset = _bbox_offset[select_mask]
        select_boxes = pnet_boxes[select_mask]

        boxes = self.__back_cal_boxs_ro(select_boxes, select_bbox_offset, select_cls)


        return utils.nms(boxes, RNET_NMS_THRESHOLD)

    def __onet_detect(self, image, rnet_boxes):
        # # 1.创建候选框列表
        # boxes = []

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
        cls_output, bbox_offset_output = self.onet(img_dataset)

        # 4.先通过置信度阈值过滤一部分；剩余部分计算对应的候选框，并添加到候选框列表中
        _cls, _bbox_offset = cls_output.cpu().detach().numpy(), bbox_offset_output.cpu().detach().numpy()
        select_mask = _cls[:, 0] > ONET_CONF_THRESHOLD
        select_cls = _cls[select_mask]
        select_bbox_offset = _bbox_offset[select_mask]
        select_boxes = rnet_boxes[select_mask]
        # 添加到候选框列表中
        boxes = self.__back_cal_boxs_ro(select_boxes, select_bbox_offset, select_cls)

        # 5.最后，通过非极大值抑制排除部分的候选框，并返回新的候选框列表
        return utils.nms(boxes, ONET_NMS_THRESHOLD, isMin=True)

    # r和o的特征图反算，返回候选框
    def __back_cal_boxs_ro(self, select_boxes, select_bbox_offset, select_cls):
        _x1, _y1, _x2, _y2 = select_boxes[:, 0], select_boxes[:, 1], select_boxes[:, 2], select_boxes[:, 3]
        _w, _h = _x2 - _x1, _y2 - _y1
        # 图片resize不改变偏移量
        x1 = _x1 + _w * select_bbox_offset[:, 0]
        y1 = _y1 + _h * select_bbox_offset[:, 1]
        x2 = _x2 + _w * select_bbox_offset[:, 2]
        y2 = _y2 + _h * select_bbox_offset[:, 3]
        # 返回候选框
        return numpy.array([x1, y1, x2, y2, select_cls.squeeze(axis=1)]).T

    def __call2__(self, image):
        pnet_start = time.time()
        # 1.将image放到P网络检测，得到经过nms后的候选框pnet_boxes
        pnet_boxes = self.__pnet_detect2(image)
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

        # # 打印网络检测时间
        # print("total:", onet_end - pnet_start, "pnet:", rnet_start - pnet_start, "rnet:", onet_start - rnet_start,
        #       "onet:", onet_end - onet_start)

        return onet_boxes

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

            cls_output, bbox_offset_output = self.pnet(img_data)

            # 4.先通过置信度阈值过滤一部分；剩余部分通过"特征图反算"得到候选框，并加入到候选框列表boxes中
            # cls_output: (1, 1, h, w)，bbox_offset_output: (1, 4, h, w)
            cls = cls_output[0][0].cpu().detach().numpy()
            bbox_offset = bbox_offset_output[0].cpu().detach().numpy()
            # 根据置信度阈值筛选
            index = numpy.where(cls > PNET_CONF_THRESHOLD)
            idxs = numpy.stack((index[0], index[1]), axis=1)
            # 通过"特征图反算"得到候选框，并添加到候选框列表中
            if idxs.shape[0] > 0:
                boxes.append(
                    self.__back_cal_boxs(idxs, cls[idxs[:, 0], idxs[:, 1]], bbox_offset[:, idxs[:, 0], idxs[:, 1]],
                                         scale))

            # 图片缩放，控制循环条件
            scale *= PYRAMID_SCALE_COEFFICIENT
            _w, _h = int(img_w * scale), int(img_h * scale)
            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)
        # 转换为[[x1, y1, x2, y2, cls], [x1, y1, x2, y2, cls], ...]结构
        if len(boxes) > 0:
            boxes = numpy.concatenate(boxes, axis=1).T
        else:
            boxes = numpy.array(boxes)

        # 5.最后，通过非极大值抑制排除部分的候选框，并返回新的候选框列表
        return utils.nms(boxes, PNET_NMS_THRESHOLD)


if __name__ == '__main__':

    detector = Detector(r"D:\MTCNN\MTCNN-Pytorch\result\other\pnet.t",
                        r"D:\MTCNN\MTCNN-Pytorch\result\other\rnet.t",
                        r"D:\MTCNN\MTCNN-Pytorch\result\other\onet.t")



    test_image_path = r"D:\MTCNN\data\MTCNN-Pytorch\detect_img"
    for filename in os.listdir(test_image_path):
        img_path = os.path.join(test_image_path, filename)

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        onet_boxes = detector(img)

        for box in onet_boxes:
            x1, y1, x2, y2, cls = box[0], box[1], box[2], box[3], box[4]


            print("bbox:", (x1, y1, x2, y2), "conf:", cls)

            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)

            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        plt.imshow(img)
        plt.show()
