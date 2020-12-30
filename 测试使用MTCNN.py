"""
使用mtcnn检测图片、视频、摄像头
---
视频采集: cv2.VideoCapture()
1) cv2.VideoCapture() 创建VideoCapture视频采集对象
    VideoCapture是用于从视频文件、图片序列、摄像头捕获视频的类。
    参数是0，表示打开笔记本的前置摄像头。
    参数是1，表示打开笔记本的后置摄像头。
    参数是视频文件路径则采集相应视频。
2) ret, frame = cap.read() 按帧读取视频，一帧就是一张图片。
    ret是bool类型，表示是否成功获取帧。如果视频读到结尾，那么ret就为False。
    frame 是每一帧的图像，是一个三维矩阵numpy.ndarray。
3) cap.release() 释放视频资源
"""
import os
import cv2
import numpy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from optimize_detect import Detector
# 查看时间
import time


# 1.检测图片
def detect_img(detector, img_dir):
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        onet_boxes = detector(img)
        for box in onet_boxes:
            x1, y1, x2, y2, cls = box[0], box[1], box[2], box[3], box[4]
            # print("bbox:", (x1, y1, x2, y2), "conf:", cls)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        plt.imshow(img)
        plt.show()


# 2.检测视频
def detect_video(detector, url):
    # 创建视频流采集对象VideoCapture
    cap = cv2.VideoCapture(url)

    while True:
        start = time.time()

        # 按帧读取视频
        ret, frame = cap.read()  # cv2读取的图片是BGR模式
        if not ret:
            continue

        # MTCNN人脸检测
        mtcnn_start = time.time()
        boxes = detector(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        mtcnn_end = time.time()

        # 画框并展示
        for box in boxes:
            # x1, y1, x2, y2, cls = box[0], box[1], box[2], box[3], box[4]
            # print("bbox:", (x1, y1, x2, y2), "conf:", cls)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)  # 创建一个窗口，可调整大小
        # cv2.resizeWindow('frame', 600, 400)  # 修改窗口大小，宽600，高400
        cv2.imshow('frame', frame)  # 展示视频图片

        # 按'q'键退出循环
        # 等价于(cv2.waitKey(41) & 0xFF) == ord('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()
        print(f"MTCNN检测耗时:{mtcnn_end - mtcnn_start}", f"总耗时:{end - start}")

    # 3.释放VideoCapture对象
    cap.release()
    cv2.destroyAllWindows()  # 销毁所有窗口


if __name__ == '__main__':
    pass
    # mtcnn侦测器
    detector1 = Detector(r"C:\Users\zxm\Desktop\MTCNN-Pytorch\result\other\pnet.t",
                         r"C:\Users\zxm\Desktop\MTCNN-Pytorch\result\other\rnet.t",
                         r"C:\Users\zxm\Desktop\MTCNN-Pytorch\result\other\onet.t")

    # # 1.检测图片
    # img_dir1 = r"D:\MTCNN\data\MTCNN-Pytorch\simple\data\test"
    # detect_img(detector1, img_dir1)

    # 2.检测视频、摄像头
    url1 = 0  # 前置摄像头
    # url1 = "http://ivi.bupt.edu.cn/hls/cctv1hd.m3u8"  # 视频流地址
    # url1 = "D:/迅雷下载/诛仙.mkv"  # 视频文件
    detect_video(detector1, url1)
