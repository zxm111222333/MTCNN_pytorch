# 定义MTCNN的P、R、O神经网络模型
"""
1、卷积核的stride=1，MaxPool池化的stride=2。
2、统一使用PReLU作为卷积层和全连接层(不包括输出层)后的非线性激活函数
"""
import torch
from torch import nn


# simple or normal------
class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.PReLU(),
            # nn.BatchNorm2d(10),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(10, 16, 3),
            nn.PReLU(),
            # nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3),
            nn.PReLU(),
            # nn.BatchNorm2d(32)
        )
        self.face_classification_output_layer = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        self.bbox_regression_output_layer = nn.Sequential(
            nn.Conv2d(32, 4, 1)
        )
        self.face_landmark_localization_output_layer = nn.Sequential(
            nn.Conv2d(32, 10, 1)
        )

    def forward(self, x):
        h = self.layers(x)
        # print(h.shape)
        face_classification_output = self.face_classification_output_layer(h)
        bbox_regression_output = self.bbox_regression_output_layer(h)
        face_landmark_localization_output = self.face_landmark_localization_output_layer(h)
        return face_classification_output, bbox_regression_output, face_landmark_localization_output


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 28, 3),
            nn.PReLU(),
            # nn.BatchNorm2d(28),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(28, 48, 3),
            nn.PReLU(),
            # nn.BatchNorm2d(48),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(48, 64, 2),
            nn.PReLU(),
            # nn.BatchNorm2d(64)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(),
            # nn.BatchNorm1d(128)
        )
        self.face_classification_output_layer = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.bbox_regression_output_layer = nn.Sequential(
            nn.Linear(128, 4)
        )
        self.face_landmark_localization_output_layer = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        h = self.layers(x)
        # print(h.shape)
        h = h.reshape(-1, 64 * 3 * 3)
        # print(h.shape)
        h = self.fc_layer(h)
        # print(h.shape)
        face_classification_output = self.face_classification_output_layer(h)
        bbox_regression_output = self.bbox_regression_output_layer(h)
        face_landmark_localization_output = self.face_landmark_localization_output_layer(h)
        return face_classification_output, bbox_regression_output, face_landmark_localization_output


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.PReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(32, 64, 3),
            nn.PReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(64, 64, 3),
            nn.PReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 2),
            nn.PReLU(),
            # nn.BatchNorm2d(128)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(),
            # nn.BatchNorm1d(256)
        )
        self.face_classification_output_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.bbox_regression_output_layer = nn.Sequential(
            nn.Linear(256, 4)
        )
        self.face_landmark_localization_output_layer = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        h = self.layers(x)
        # print(h.shape)
        h = h.reshape(-1, 128 * 3 * 3)
        h = self.fc_layer(h)
        # print(h.shape)
        face_classification_output = self.face_classification_output_layer(h)
        bbox_regression_output = self.bbox_regression_output_layer(h)
        face_landmark_localization_output = self.face_landmark_localization_output_layer(h)
        return face_classification_output, bbox_regression_output, face_landmark_localization_output


# other------
class Pnet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.PReLU(),

            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU())
        self.confid_layer = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1),
            nn.Sigmoid())
        self.offset_layer = nn.Sequential(
            nn.Conv2d(32, 4, 1, 1))
        self.keypoints_layer = nn.Sequential(
            nn.Conv2d(32, 10, 1, 1))

    def forward(self, x):
        h = self.layers(x)

        confidence = self.confid_layer(h)
        offset = self.offset_layer(h)
        keypoints = self.keypoints_layer(h)
        return confidence, offset, keypoints


class Rnet2(nn.Module):
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


class Onet2(nn.Module):
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


# bn------
class PNet3(nn.Module):
    def __init__(self):
        super(PNet3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.PReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(10, 16, 3),
            nn.PReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3),
            nn.PReLU(),
            nn.BatchNorm2d(32)
        )
        self.face_classification_output_layer = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        self.bbox_regression_output_layer = nn.Sequential(
            nn.Conv2d(32, 4, 1)
        )
        self.face_landmark_localization_output_layer = nn.Sequential(
            nn.Conv2d(32, 10, 1)
        )

    def forward(self, x):
        h = self.layers(x)
        # print(h.shape)
        face_classification_output = self.face_classification_output_layer(h)
        bbox_regression_output = self.bbox_regression_output_layer(h)
        face_landmark_localization_output = self.face_landmark_localization_output_layer(h)
        return face_classification_output, bbox_regression_output, face_landmark_localization_output


class RNet3(nn.Module):
    def __init__(self):
        super(RNet3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 28, 3),
            nn.PReLU(),
            nn.BatchNorm2d(28),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(28, 48, 3),
            nn.PReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(48, 64, 2),
            nn.PReLU(),
            nn.BatchNorm2d(64)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128)
        )
        self.face_classification_output_layer = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.bbox_regression_output_layer = nn.Sequential(
            nn.Linear(128, 4)
        )
        self.face_landmark_localization_output_layer = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        h = self.layers(x)
        # print(h.shape)
        h = h.reshape(-1, 64 * 3 * 3)
        # print(h.shape)
        h = self.fc_layer(h)
        # print(h.shape)
        face_classification_output = self.face_classification_output_layer(h)
        bbox_regression_output = self.bbox_regression_output_layer(h)
        face_landmark_localization_output = self.face_landmark_localization_output_layer(h)
        return face_classification_output, bbox_regression_output, face_landmark_localization_output


class ONet3(nn.Module):
    def __init__(self):
        super(ONet3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(32, 64, 3),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(64, 64, 3),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 2),
            nn.PReLU(),
            nn.BatchNorm2d(128)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256)
        )
        self.face_classification_output_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.bbox_regression_output_layer = nn.Sequential(
            nn.Linear(256, 4)
        )
        self.face_landmark_localization_output_layer = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        h = self.layers(x)
        # print(h.shape)
        h = h.reshape(-1, 128 * 3 * 3)
        h = self.fc_layer(h)
        # print(h.shape)
        face_classification_output = self.face_classification_output_layer(h)
        bbox_regression_output = self.bbox_regression_output_layer(h)
        face_landmark_localization_output = self.face_landmark_localization_output_layer(h)
        return face_classification_output, bbox_regression_output, face_landmark_localization_output


if __name__ == '__main__':
    # # PNet
    # x = torch.randn(100, 3, 14, 14)
    # # x = torch.randn(100, 3, 12, 12)
    # # x = torch.randn(2, 3, 12, 12)
    # p_net = PNet()
    # y = p_net(x)
    # print(y[0].shape)
    # print(y[1].shape)
    # print(y[2].shape)
    # # (N,C,H,W)-->(N,V)
    # t = torch.squeeze(torch.squeeze(y[0], dim=2), dim=2)
    # print(t, t.shape)
    # m = y[0].reshape(y[0].shape[0], -1)
    # print(m, m.shape)
    # n = y[0].reshape(-1, 1)
    # print(n, n.shape)

    # # RNet
    # x = torch.randn(100, 3, 24, 24)
    # r_net = RNet()
    # y = r_net(x)
    # print(y[0].shape)
    # print(y[1].shape)
    # print(y[2].shape)

    # ONet
    x = torch.randn(100, 3, 48, 48)
    o_net = ONet()
    y = o_net(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)