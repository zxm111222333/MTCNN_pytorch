# 训练RNet
from train import Trainer
from nets import RNet

if __name__ == '__main__':
    # normal
    trainer = Trainer(r"D:\MTCNN\data\MTCNN-Pytorch\normal\gen\CelebA\24",
                      512, RNet(),
                      r"result/normal/checkpoint/r",
                      r"result/normal/logs/r",
                      (1, 0.5, 0.5), True)
    trainer()
