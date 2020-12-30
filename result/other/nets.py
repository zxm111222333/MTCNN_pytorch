from torch import nn
import torch

class Pnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.PReLU(),

            # nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),  # conv1
            # nn.PReLU(),  # prelu1
            # nn.MaxPool2d(kernel_size=3, stride=2),  # pool1；conv1里的填充在此处操作，效果更好★

            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU())
        self.confid_layer = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1),
            nn.Sigmoid())
        self.offset_layer = nn.Sequential(
            nn.Conv2d(32,4,1,1))
        self.keypoints_layer = nn.Sequential(
            nn.Conv2d(32,10,1,1))

    def forward(self, x):
        h = self.layers(x)
        # convd1 = nn.Conv2d(32, 1, 1, 1)
        # out1 = convd1(h).reshape(-1, 1)
        # confidence = torch.sigmoid(out1)
        # ret1 = nn.functional.sigmoid(out1)
        # convd2 = nn.Conv2d(32, 4, 1, 1)
        # offset = convd2(h).reshape(-1, 4)
        # convd3 = nn.Conv2d(32, 10, 1, 1)
        # keypoints = convd3(h).reshape(-1, 10)

        confidence = self.confid_layer(h)
        offset = self.offset_layer(h)
        keypoints = self.keypoints_layer(h)
        return confidence, offset, keypoints


class Rnet(nn.Module):
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

        keypoints = self.keypoints_layer(h)
        return confidence, offset, keypoints


class Onet(nn.Module):
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
        keypoints = self.keypoints_layer(h)
        return confidence, offset, keypoints


if __name__ == '__main__':
    # net = Pnet()
    # x = torch.randn(5, 3, 12, 12)
    # print(net(x)[1].shape,net(x)[2].shape)

    net = Rnet()
    x = torch.randn(5, 3, 24, 24)
    print(net(x)[1].shape,net(x)[2].shape)

    # net = Onet()
    # x = torch.randn(5, 3, 48, 48)
    # y = net(x)
    # print(y)
