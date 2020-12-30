# 生成样本图片及对应的标签
"""
流程：
1.下载CelebA数据集到本地，包括图片和标签数据。
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    百度网盘：
        https://pan.baidu.com/s/1CRxxhoQ97A5qbsKO7iaAJg#list/path=%2F
        密码: rp0s
2.新建生成样本的保存目录，形如：
    12/positive、12/negative、12/part
    24/positive、24/negative、24/part
    48/positive、48/negative、48/part
3.读取原始标签数据，并获取对应的图片信息。
4.对于每张图片，随机切割生成positive、negative和part样本，并保存新图片和新标签数据。
"""
import os
from PIL import Image
import numpy
from tools import utils
import traceback


class GenData:
    def __init__(self, src_img_path, src_anno_file, gen_save_path, size):
        super(GenData, self).__init__()
        self.src_img_path = src_img_path
        self.src_anno_file = src_anno_file
        self.gen_save_path = gen_save_path
        self.size = size

    def __call__(self):
        print(f"生成{self.size}X{self.size}的图片:")
        # 1.样本保存目录
        # "样本图片"的保存路径
        positive_img_path = os.path.join(self.gen_save_path, str(self.size), "positive")
        part_img_path = os.path.join(self.gen_save_path, str(self.size), "part")
        negative_img_path = os.path.join(self.gen_save_path, str(self.size), "negative")
        for img_path in [positive_img_path, part_img_path, negative_img_path]:
            if not os.path.exists(img_path):
                os.makedirs(img_path)
        # “样本标签”的保存文件名
        positive_anno_filename = os.path.join(self.gen_save_path, str(self.size), "positive.txt")
        part_anno_filename = os.path.join(self.gen_save_path, str(self.size), "part.txt")
        negative_anno_filename = os.path.join(self.gen_save_path, str(self.size), "negative.txt")

        # 2.生成样本图片及标签
        positive_counter = 0
        part_counter = 0
        negative_counter = 0
        # 对于文件操作，一般使用try
        try:
            # 打开要写入的文件
            positive_anno_file = open(positive_anno_filename, mode="a")
            part_anno_file = open(part_anno_filename, mode="a")
            negative_anno_file = open(negative_anno_filename, mode="a")

            # 逐行读取原始标签文件
            for i, line in enumerate(open(self.src_anno_file)):
                if i < 2:
                    continue  # 跳过文件头两行
                # print(i, line.strip().split())

                # 对于文件操作，一般使用try
                try:
                    # 2.1得到每一行的标签数据: 文件名、x1、y1、w、h...
                    tag_data = line.strip().split()
                    # print(tag_data)
                    img_name = tag_data[0]
                    x1 = float(tag_data[1])
                    y1 = float(tag_data[2])
                    w = float(tag_data[3])
                    h = float(tag_data[4])
                    lefteye_x = float(tag_data[5])
                    lefteye_y = float(tag_data[6])
                    righteye_x = float(tag_data[7])
                    righteye_y = float(tag_data[8])
                    nose_x = float(tag_data[9])
                    nose_y = float(tag_data[10])
                    leftmouth_x = float(tag_data[11])
                    leftmouth_y = float(tag_data[12])
                    rightmouth_x = float(tag_data[13])
                    rightmouth_y = float(tag_data[14])
                    # 过滤，去除不符合条件的坐标
                    if x1 < 0 or y1 < 0 or w < 0 or h < 0 or lefteye_x < 0 or lefteye_y < 0 or righteye_x < 0 or righteye_y < 0 or nose_x < 0 or nose_y < 0 or leftmouth_x < 0 or leftmouth_y < 0 or rightmouth_x < 0 or rightmouth_y < 0 or max(
                            w, h) < 40:
                        continue
                    # 标签框、中心点坐标
                    x2 = x1 + w
                    y2 = y1 + h
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    boxes = numpy.array([[x1, y1, x2, y2]])

                    # 2.2对于每一张原始图片, 生成相应数量的positive、part、negative样本
                    with Image.open(os.path.join(self.src_img_path, img_name)) as img:
                        # 获取原始图片的宽和高
                        img_w, img_h = img.size

                        # 生成positive、part、negative样本(每一张图片生成positive、part、negative各位1.5、1.5、3张左右)
                        for i in range(5):
                            # 新中心点: 上下左右偏移范围20%
                            cx_ = cx + numpy.random.randint(-int(w * 0.2), numpy.ceil(w * 0.2))
                            cy_ = cy + numpy.random.randint(-int(h * 0.2), numpy.ceil(h * 0.2))
                            # 新边长: 介于0.8*min(w,h)~1.25*max(m,h)之间
                            side_len = numpy.random.randint(int(min(w, h) * 0.8), numpy.ceil(max(w, h) * 1.25))

                            # 新标签框
                            x1_ = cx_ - side_len / 2
                            y1_ = cy_ - side_len / 2
                            x2_ = x1_ + side_len
                            y2_ = y1_ + side_len
                            # 过滤，去除超出截取范围的图片
                            if x1_ < 0 or y1_ < 0 or x2_ > img_w or y2_ > img_h:
                                continue
                            crop_box = numpy.array([x1_, y1_, x2_, y2_])

                            # 计算坐标的偏移量
                            offset_x1 = (x1 - x1_) / side_len
                            offset_y1 = (y1 - y1_) / side_len
                            offset_x2 = (x2 - x2_) / side_len
                            offset_y2 = (y2 - y2_) / side_len
                            offset_lefteye_x = (lefteye_x - x1_) / side_len
                            offset_lefteye_y = (lefteye_y - y1_) / side_len
                            offset_righteye_x = (righteye_x - x1_) / side_len
                            offset_righteye_y = (righteye_y - y1_) / side_len
                            offset_nose_x = (nose_x - x1_) / side_len
                            offset_nose_y = (nose_y - y1_) / side_len
                            offset_leftmouth_x = (leftmouth_x - x1_) / side_len
                            offset_leftmouth_y = (leftmouth_y - y1_) / side_len
                            offset_rightmouth_x = (rightmouth_x - x1_) / side_len
                            offset_rightmouth_y = (rightmouth_y - y1_) / side_len

                            # 剪切下图片，并进行缩放-->12*12, 24*24, 48*48
                            crop_img = img.crop(crop_box)
                            resize_img = crop_img.resize((self.size, self.size), Image.ANTIALIAS)
                            # 计算iou，并根据iou进行样本分类保存
                            iou = utils.iou(crop_box, boxes)[0]
                            # print(iou)
                            if iou > 0.8:
                                # 写入标签数据
                                positive_anno_file.write(
                                    "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                        positive_counter + 1, 1, offset_x1, offset_y1, offset_x2, offset_y2,
                                        offset_lefteye_x, offset_lefteye_y, offset_righteye_x, offset_righteye_y,
                                        offset_nose_x, offset_nose_y, offset_leftmouth_x, offset_leftmouth_y,
                                        offset_rightmouth_x, offset_rightmouth_y))
                                positive_anno_file.flush()
                                # 保存图片
                                resize_img.save(
                                    os.path.join(positive_img_path, "{0}.jpg".format(positive_counter + 1)))
                                positive_counter += 1
                                print("图片大小:", self.size, "正样本数量:", positive_counter)
                            elif iou > 0.5:
                                # 写入标签数据
                                part_anno_file.write(
                                    "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                        part_counter + 1, 2, offset_x1, offset_y1, offset_x2, offset_y2,
                                        offset_lefteye_x, offset_lefteye_y, offset_righteye_x, offset_righteye_y,
                                        offset_nose_x, offset_nose_y, offset_leftmouth_x, offset_leftmouth_y,
                                        offset_rightmouth_x, offset_rightmouth_y))
                                part_anno_file.flush()
                                # 保存图片
                                resize_img.save(os.path.join(part_img_path, "{0}.jpg".format(part_counter + 1)))
                                part_counter += 1
                                print("图片大小:", self.size, "部分样本数量:", part_counter)
                            elif iou < 0.05:
                                # 写入标签数据
                                negative_anno_file.write(
                                    "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                        negative_counter + 1,
                                        0))
                                negative_anno_file.flush()
                                # 保存图片
                                resize_img.save(
                                    os.path.join(negative_img_path, "{0}.jpg".format(negative_counter + 1)))
                                negative_counter += 1
                                print("图片大小:", self.size, "负样本数量:", negative_counter)

                        # 补充生成negative样本
                        for i in range(7):
                            # 新边长
                            if int(min(img_w, img_h) / 2) < self.size:
                                continue
                            side_len = numpy.random.randint(self.size, int(min(img_w, img_h) / 2) + 1)
                            # 新标签框
                            x1_ = numpy.random.randint(0, img_w - side_len)
                            y1_ = numpy.random.randint(0, img_h - side_len)
                            x2_ = x1_ + side_len
                            y2_ = y1_ + side_len
                            crop_box = numpy.array([x1_, y1_, x2_, y2_])

                            iou = utils.iou(crop_box, boxes)[0]
                            if iou < 0.05:
                                crop_img = img.crop(crop_box)
                                resize_img = crop_img.resize((self.size, self.size), Image.ANTIALIAS)
                                negative_anno_file.write(
                                    "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                        negative_counter + 1,
                                        0))
                                negative_anno_file.flush()
                                resize_img.save(
                                    os.path.join(negative_img_path, "{0}.jpg".format(negative_counter + 1)))
                                negative_counter += 1
                                print("图片大小:", self.size, "负样本数量:", negative_counter)
                except Exception as e:
                    # print(e)
                    traceback.print_exc()

        finally:
            # 关闭文件
            positive_anno_file.close()
            part_anno_file.close()
            negative_anno_file.close()
        print("图片大小:", self.size, "样本总数:", positive_counter + part_counter + negative_counter, "正样本数量:",
              positive_counter, "部分样本数量:", part_counter, "负样本数量:", negative_counter)


if __name__ == '__main__':
    pass
    # # 1.代码测试
    # # 原始图片路径和标签文件
    # src_img_path1 = r"D:\MTCNN\data\MTCNN-Pytorch\test\img"
    # src_anno_file1 = r"D:\MTCNN\data\MTCNN-Pytorch\test\ano\1.txt"
    # # 生成的样本图片及标签的保存路径
    # gen_save_path1 = r"D:\MTCNN\data\MTCNN-Pytorch\test\gen\CelebA"
    #
    # for size in [12, 24, 48]:
    #     genData = GenData(src_img_path1, src_anno_file1, gen_save_path1, size)
    #     genData()

    # # 2.simple: 简单版本1W张图片数据生成
    # # 原始图片路径和标签文件
    # src_img_path1 = r"D:\MTCNN\data\MTCNN-Pytorch\simple\src\img_celeba"
    # src_anno_file1 = r"D:\MTCNN\data\MTCNN-Pytorch\simple\src\list_merge_celeba.txt"
    # # 生成的样本图片及标签的保存路径
    # gen_save_path1 = r"D:\MTCNN\data\MTCNN-Pytorch\simple\gen\CelebA"
    #
    # for size in [12, 24, 48]:
    #     genData = GenData(src_img_path1, src_anno_file1, gen_save_path1, size)
    #     genData()

    # # 3.normal: 所有celeba数据生成
    # # 原始图片路径和标签文件
    # src_img_path1 = r"D:\MTCNN\data\MTCNN-Pytorch\normal\src\CelebA\Img\img_celeba.7z\img_celeba"
    # src_anno_file1 = r"D:\MTCNN\data\MTCNN-Pytorch\normal\src\list_merge_celeba.txt"
    # # 生成的样本图片及标签的保存路径
    # gen_save_path1 = r"D:\MTCNN\data\MTCNN-Pytorch\normal\gen\CelebA"
    #
    # for size in [12, 24, 48]:
    #     genData = GenData(src_img_path1, src_anno_file1, gen_save_path1, size)
    #     genData()
