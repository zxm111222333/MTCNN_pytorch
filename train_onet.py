# 训练ONet
from train import Trainer
from nets import ONet
import random

if __name__ == '__main__':
    trainer = Trainer(r"D:\MTCNN\data\MTCNN-Pytorch\normal\gen\CelebA\48",
                      512, ONet(),
                      r"result/normal/checkpoint/o",
                      r"result/normal/logs/o",
                      (1, 0.5, 1), True)


    trainer()
