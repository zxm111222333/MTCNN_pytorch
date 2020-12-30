# 训练PNet
from train import Trainer
from nets import PNet

if __name__ == '__main__':
    # normal
    trainer = Trainer(r"D:\MTCNN\data\MTCNN-Pytorch\normal\gen\CelebA\12",
                      512, PNet(),
                      r"result/normal/checkpoint/p",
                      r"result/normal/logs/p",
                      (1, 0.5, 0.5), True)



    trainer()
