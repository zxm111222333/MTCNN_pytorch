# 创建训练器
from sampling import FaceDataset
from nets import *
import torch
from torch.utils.data import DataLoader
# tensorboard可视化
from torch.utils.tensorboard import SummaryWriter
# 查看时间和进度
import time
from tqdm import tqdm


class Trainer:
    def __init__(self, dataset_path, batch_size, net, save_path, log_path, weight_coefficients, is_cuda=True):
        super(Trainer, self).__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.net = net
        self.save_path = save_path
        self.log_path = log_path
        self.weight_coefficients = weight_coefficients  # 计算损失的权重系数，形如(1, 0.5, 0.5)、(1, 0.5, 1)
        self.is_cuda = is_cuda

        # 1.加载数据集
        # 训练集
        self.train_dataset = FaceDataset(self.dataset_path)
        self.train_dataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        # 测试(验证)集
        self.test_dataset = FaceDataset(self.dataset_path, False)
        self.test_dataLoader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # 2.创建模型
   
        # 将数据移到gpu设备上
        if self.is_cuda:
            self.net.cuda()

        # 3.定义优化器
        self.opt = torch.optim.Adam(self.net.parameters())

        # 4.定义损失函数
        # 分类损失函数: 置信度(人脸分类识别)
        self.cls_loss_fun = torch.nn.BCELoss()  # Sigmoid交叉熵
        # 回归损失函数: 偏移量(边界框回归、人脸关键点定位)
        self.reg_loss_fun = torch.nn.MSELoss()

        # 5.tensorboard可视化
        self.summaryWriter = SummaryWriter(log_dir=self.log_path)

    def __call__(self):
        start = time.time()

        for epoch in range(10000):
            # 1.训练
            # 每一次轮询的总损失、平均损失
            train_loss_sum, train_loss_avg = 0., 0.
            train_cls_loss_sum, train_cls_loss_avg = 0., 0.
            train_bbox_loss_sum, train_bbox_loss_avg = 0., 0.
            train_landmark_loss_sum, train_landmark_loss_avg = 0., 0.
            for i, (img_data, cls, bbox_offset, landmark_offset) in enumerate(tqdm(self.train_dataLoader)):
                # 将数据移到gpu设备上
                if self.is_cuda:
                    img_data = img_data.cuda()
                    cls = cls.cuda()
                    bbox_offset = bbox_offset.cuda()
                    landmark_offset = landmark_offset.cuda()

                cls_output, bbox_offset_output, landmark_offset_output = self.net(img_data)
                # 若是PNet，将结果[n, c, 1, 1]转换为[n, v]
                cls_output = cls_output.reshape(cls_output.shape[0], -1)
                bbox_offset_output = bbox_offset_output.reshape(bbox_offset_output.shape[0], -1)
                landmark_offset_output = landmark_offset_output.reshape(landmark_offset_output.shape[0], -1)

                # 计算loss
                # 1.人脸识别分类损失cls_loss: 正样本cls=1和负样本cls=0
                cls_index = torch.where(cls < 2)[0]
                select_cls = cls[cls_index]
                select_cls_output = cls_output[cls_index]
                cls_loss = self.cls_loss_fun(select_cls_output, select_cls)
                # 2.边框回归损失bbox_loss: 正样本cls=1和部分样本cls=2
                bbox_index = torch.where(cls > 0)[0]
                select_bbox_offset = bbox_offset[bbox_index]
                select_bbox_offset_output = bbox_offset_output[bbox_index]
                bbox_loss = self.reg_loss_fun(select_bbox_offset_output, select_bbox_offset)
                # 3.人脸关键点回归损失landmark_loss: 关键点样本(本次测试用正样本cls=1和部分样本cls=2作为关键点样本)
                landmark_index = torch.where(cls > 0)[0]
                select_landmark_offset = landmark_offset[landmark_index]
                select_landmark_offset_output = landmark_offset_output[landmark_index]
                landmark_loss = self.reg_loss_fun(select_landmark_offset_output, select_landmark_offset)
                # 总损失
                loss = self.weight_coefficients[0] * cls_loss + self.weight_coefficients[1] * bbox_loss + \
                       self.weight_coefficients[2] * landmark_loss

                # 清空梯度
                self.opt.zero_grad()
                # 反向传播
                loss.backward()
                # 更新梯度
                self.opt.step()

                # 计算总损失
                train_cls_loss_sum += cls_loss.cpu().item()
                train_bbox_loss_sum += bbox_loss.cpu().item()
                train_landmark_loss_sum += landmark_loss.cpu().item()
                train_loss_sum += loss.cpu().item()
            # 计算平均损失
            train_cls_loss_avg = train_cls_loss_sum / len(self.train_dataLoader)
            train_bbox_loss_avg = train_bbox_loss_sum / len(self.train_dataLoader)
            train_landmark_loss_avg = train_landmark_loss_sum / len(self.train_dataLoader)
            train_loss_avg = train_loss_sum / len(self.train_dataLoader)

            # 2.测试(验证)
            # 每一次轮询的总损失、平均损失
            test_loss_sum, test_loss_avg = 0., 0.
            test_cls_loss_sum, test_cls_loss_avg = 0., 0.
            test_bbox_loss_sum, test_bbox_loss_avg = 0., 0.
            test_landmark_loss_sum, test_landmark_loss_avg = 0., 0.
            for i, (img_data, cls, bbox_offset, landmark_offset) in enumerate(tqdm(self.test_dataLoader)):
                # 将数据移到gpu设备上
                if self.is_cuda:
                    img_data = img_data.cuda()
                    cls = cls.cuda()
                    bbox_offset = bbox_offset.cuda()
                    landmark_offset = landmark_offset.cuda()

                cls_output, bbox_offset_output, landmark_offset_output = self.net(img_data)
                # 若是PNet，将结果[n, c, 1, 1]转换为[n, v]
                cls_output = cls_output.reshape(cls_output.shape[0], -1)
                bbox_offset_output = bbox_offset_output.reshape(bbox_offset_output.shape[0], -1)
                landmark_offset_output = landmark_offset_output.reshape(landmark_offset_output.shape[0], -1)

                # 计算loss
                # 1.人脸识别分类损失cls_loss: 正样本cls=1和负样本cls=0
                cls_index = torch.where(cls < 2)[0]
                select_cls = cls[cls_index]
                select_cls_output = cls_output[cls_index]
                cls_loss = self.cls_loss_fun(select_cls_output, select_cls)
                # 2.边框回归损失bbox_loss: 正样本cls=1和部分样本cls=2
                bbox_index = torch.where(cls > 0)[0]
                select_bbox_offset = bbox_offset[bbox_index]
                select_bbox_offset_output = bbox_offset_output[bbox_index]
                bbox_loss = self.reg_loss_fun(select_bbox_offset_output, select_bbox_offset)
                # 3.人脸关键点回归损失landmark_loss: 关键点样本(本次测试用正样本cls=1和部分样本cls=2作为关键点样本)
                landmark_index = torch.where(cls > 0)[0]
                select_landmark_offset = landmark_offset[landmark_index]
                select_landmark_offset_output = landmark_offset_output[landmark_index]
                landmark_loss = self.reg_loss_fun(select_landmark_offset_output, select_landmark_offset)
                # 总损失
                loss = self.weight_coefficients[0] * cls_loss + self.weight_coefficients[1] * bbox_loss + \
                       self.weight_coefficients[2] * landmark_loss

                # 清空梯度
                self.opt.zero_grad()
                # 反向传播
                loss.backward()
                # 更新梯度
                self.opt.step()

                # 计算总损失
                test_cls_loss_sum += cls_loss.cpu().item()
                test_bbox_loss_sum += bbox_loss.cpu().item()
                test_landmark_loss_sum += landmark_loss.cpu().item()
                test_loss_sum += loss.cpu().item()
            # 计算平均损失
            test_cls_loss_avg = test_cls_loss_sum / len(self.test_dataLoader)
            test_bbox_loss_avg = test_bbox_loss_sum / len(self.test_dataLoader)
            test_landmark_loss_avg = test_landmark_loss_sum / len(self.test_dataLoader)
            test_loss_avg = test_loss_sum / len(self.test_dataLoader)

            # 3.更多操作
            # 1）打印loss信息
            print(epoch, "train_loss:", train_loss_avg, "train_cls_loss:", train_cls_loss_avg, "train_bbox_loss:",
                  train_bbox_loss_avg, "train_landmark_loss:", train_landmark_loss_avg)
            print(epoch, "test_loss:", test_loss_avg, "test_cls_loss:", test_cls_loss_avg, "test_bbox_loss:",
                  test_bbox_loss_avg, "test_landmark_loss:", test_landmark_loss_avg)
            # 2）保存模型
            torch.save(self.net.state_dict(), f"{self.save_path}/{epoch}.pt")
            # 3）tensorboard可视化
            self.summaryWriter.add_scalars("loss", {"total": train_loss_avg, "cls": train_cls_loss_avg,
                                                    "bbox": train_bbox_loss_avg, "landmark": train_landmark_loss_avg},
                                           epoch)

            # 4）查看时间和进度
            end = time.time()
            print(f"第{epoch}次轮询，共耗时{end - start}秒")
            time.sleep(0.1)


if __name__ == '__main__':
    # 训练PNet
    dataset_path1 = r"D:\MTCNN\data\MTCNN-Pytorch\normal\gen\CelebA\12"
    net1 = PNet()
    save_path1 = r"result\normal\checkpoint\p"
    batch_size1 = 5
    log_path1 = r"result\normal\logs\p"
    weight_coefficients1 = (1, 0.5, 0.5)  # p、r
    # weight_coefficients1 = (1, 0.5, 1)  # o
    is_cuda1 = True
    trainer = Trainer(dataset_path1, batch_size1, net1, save_path1, log_path1, weight_coefficients1, is_cuda1)
    trainer()
