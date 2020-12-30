# 定义MTCNN的P、R、O神经网络模型
import torch
from torch import nn


# other------
class PNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.PReLU(),

            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),

            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU()
        )
        self.confid_layer = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1),
            nn.Sigmoid()
        )
        self.offset_layer = nn.Sequential(
            nn.Conv2d(32, 4, 1, 1)
        )
        self.keypoints_layer = nn.Sequential(
            nn.Conv2d(32, 10, 1, 1))

    def forward(self, x):
        h = self.layers(x)
        confidence = self.confid_layer(h)
        offset = self.offset_layer(h)

        # keypoints = self.keypoints_layer(h)
        # return confidence, offset, keypoints
        return confidence, offset


class RNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.PReLU(),

            nn.Conv2d(28, 48, 3, 1),
            nn.MaxPool2d(3, 2),
            nn.PReLU(),

            nn.Conv2d(48, 64, 2, 1),
            nn.PReLU()
        )
        self.confid_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.offset_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(),

            nn.Linear(128, 4)
        )
        self.keypoints_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        h = self.layers(x).reshape(-1, 64 * 3 * 3)
        # print(h.shape)
        confidence = self.confid_layer(h)
        offset = self.offset_layer(h)

        # keypoints = self.keypoints_layer(h)
        # return confidence, offset, keypoints
        return confidence, offset


class ONet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.PReLU(),

            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(3, 2),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.PReLU(),

            nn.Conv2d(64, 128, 2, 1),
            nn.PReLU()
        )
        self.confid_layer = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.PReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.offset_layer = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.PReLU(),

            nn.Linear(256, 4)
        )
        self.keypoints_layer = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.PReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        h = self.layers(x).reshape(-1, 3 * 3 * 128)
        # print(h.shape)
        confidence = self.confid_layer(h)
        offset = self.offset_layer(h)


        return confidence, offset


if __name__ == '__main__':
    pass
