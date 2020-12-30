# 定义数据集
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import random


class FaceDataset(Dataset):
    def __init__(self, path, is_train=True):
        super(FaceDataset, self).__init__()
        self.path = path

        # 下面逻辑有点重复，因为之前生成图片数据的时候没有分TRAIN和TEST，所以先这样写。
        dataset = []
        dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        dataset.extend(open(os.path.join(path, "part.txt")).readlines())
        dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        # 按固定种子打乱顺序,这样每次打乱的数据都一样
        random.seed(100)
        random.shuffle(dataset)
        random.seed(50)
        random.shuffle(dataset)
        random.seed(1)

        length = len(dataset)
        if is_train:
            self.dataset = dataset[:int(0.8 * length)]
        else:
            self.dataset = dataset[int(0.8 * length):]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_list = self.dataset[index].strip().split()

        # 图片
        filename = data_list[0]
        img = Image.open(os.path.join(self.path, filename))
        img_data = transforms.ToTensor()(img)  # 自动转换为C,H,W
        # 标签: 类别+bbox偏移量+landmark偏移量
        cls = torch.Tensor([float(data_list[1])])
        bbox_offset = torch.Tensor([float(data_list[2]), float(data_list[3]), float(data_list[4]), float(data_list[5])])
        landmark_offset = torch.Tensor(
            [float(data_list[6]), float(data_list[7]), float(data_list[8]), float(data_list[9]), float(data_list[10]),
             float(data_list[11]), float(data_list[12]), float(data_list[13]), float(data_list[14]),
             float(data_list[15])])

        return img_data, cls, bbox_offset, landmark_offset


if __name__ == '__main__':
    path1 = r"data\gen\CelebA\48"
    faceDataset = FaceDataset(path1)
    print(len(faceDataset))
    # print(faceDataset[0])
    facedataloader = DataLoader(faceDataset, batch_size=10, shuffle=True)
    # facedataloader = DataLoader(faceDataset, batch_size=10, shuffle=True, drop_last=True)
    print(len(facedataloader))
    # for i, (img_data, cls, bbox_offset, landmark_offset) in enumerate(facedataloader):
    #     if i == 0:
    #         # if i == (len(facedataloader) - 1):
    #         print(img_data.shape)  # [10, 3, 48, 48]
    #         print(cls.shape, cls)  # [10, 1]
    #         print(bbox_offset.shape, bbox_offset)  # [10, 4]
    #         print(landmark_offset.shape, landmark_offset)  # [10, 10]
