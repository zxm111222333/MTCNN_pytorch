from result.other.nets import Pnet, Rnet, Onet
from result.other.utils import nms, convert_to_square
import torch, time, cv2, os
from torchvision import transforms
import numpy as np

p_confid = 0.9
p_nms = 0.5
r_confid = 0.95
r_nms = 0.5
o_confid = 0.9999
o_nms = 0.5
DEVICE = "cuda:0"


class Detector():
    def __init__(self, p_parmerter="./pnet.t", r_parmerter="./rnet.t",
                 o_parmerter="./onet.t"):
        self.pnet = Pnet().cuda()
        self.pnet.load_state_dict(torch.load(p_parmerter))
        self.rnet = Rnet().cuda()
        self.rnet.load_state_dict(torch.load(r_parmerter))
        self.onet = Onet().cuda()
        self.onet.load_state_dict(torch.load(o_parmerter))

        # if iscuda:
        #     self.pnet.to(DEVICE)
        #     self.rnet.to(DEVICE)
        #     self.onet.to(DEVICE)

    def detector(self, image):
        p_start_time = time.time()
        res_p = self.__p_detector(image)
        if res_p.shape[0] == 0:
            return np.array([])
        p_end_time = time.time()
        # print(len(res_p))
        # return res_p
        # res_p = torch.from_numpy(res_p).cuda()

        res_r = self.__r_detector(image, res_p)
        if res_r.shape[0] == 0:
            return np.array([])
        r_end_time = time.time()
        # return res_r

        res_o = self.__o_detector(image, res_r)
        o_end_time = time.time()

        p_time = p_end_time - p_start_time
        r_time = r_end_time - p_end_time
        o_time = o_end_time - r_end_time
        total_time = p_time + r_time + o_time

        print("total_time:", total_time, "p_time:", p_time, "r_time:", r_time, "o_time:", o_time)

        return res_o
        # return res_p
        # return res_r

    def __p_detector(self, image):
        res_p = np.random.rand(1, 5)
        # img = Image.open(img_path)
        img = image
        h, w, _ = img.shape
        _w, _h = w, h
        # print(w, h)
        scale = 1
        while min(h, w) > 18:
            decode = transforms.ToTensor()
            img_data = decode(img)
            img_data = torch.unsqueeze(img_data, dim=0).cuda()  #
            # print(img_data.shape)
            # exit()
            # c, h, w = img_data.shape
            # img_data = img_data.reshape(1, c, h, w)
            confidence, offset, keypoints = self.pnet(img_data)

            confidence = confidence.detach().cpu().numpy()
            offset = offset.detach().cpu().numpy()
            # keypoints = keypoints.detach().numpy()

            # res = np.hstack((offset, confidence, keypoints)).reshape(15, -1)
            # print(res.shape)
            # mask = np.where(res[4] > p_confid)[0]
            # res = res[:, mask].T

            # 计算特征图在原图上的索引
            index_h = np.where(confidence > p_confid)[2]
            index_w = np.where(confidence > p_confid)[3]

            confidence = confidence[:, :, index_h, index_w].T.reshape(-1, 1)
            # print(confidence)

            offset = offset[:, :, index_h, index_w].T.reshape(-1, 4)
            # print(offset.shape)
            # keypoints = keypoints[:, :, index_h, index_w].T.reshape(-1, 10)
            # print(keypoints.shape)
            # print(index_x.shape,index_y.shape)

            ret = self.__inverse_cal(index_w, index_h, offset, confidence, scale)
            res_p = np.concatenate((res_p, ret))
            # print(res_p.shape)
            scale = scale * 0.7
            w = int(_w * scale)
            h = int(_h * scale)
            img = cv2.resize(img, (w, h))
            # img = img.resize((w, h))
            # print(w, h)
            # img.show()
        res = res_p[1:]
        print(len(res))
        res = nms(res, p_nms)
        print(len(res))
        return res

    def __inverse_cal(self, index_x, index_y, offset, confidence, scale, stride=2, side=12):
        _x1 = (index_x * stride) / scale
        _y1 = (index_y * stride) / scale
        _x2 = (index_x * stride + side) / scale
        _y2 = (index_y * stride + side) / scale

        w = side / scale

        offset[:, 0] = offset[:, 0] * w + _x1
        offset[:, 1] = offset[:, 1] * w + _y1
        offset[:, 2] = offset[:, 2] * w + _x2
        offset[:, 3] = offset[:, 3] * w + _y2

        ret = np.hstack((offset, confidence))
        # print(ret.shape)
        # ret = ret.astype(np.int32)

        return ret

    def __r_detector(self, image, res_p):
        boxes = convert_to_square(res_p)
        # print(boxes)
        # exit()
        img_data = []
        for box in boxes:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            # print(_y1,_y2,_x1,_x2)
            # img = image.crop((_x1, _y1, _x2, _y2))
            # img = img.resize((24, 24))
            img = image[_y1:_y2, _x1:_x2]
            # print(img)
            img = cv2.resize(img, (24, 24))
            decoe = transforms.ToTensor()
            img = decoe(img)
            # print(_x1, _y1, _x2, _y2)
            # print(img)
            # exit()
            img_data.append(img)
        img_data = torch.stack(img_data).cuda()

        confid, offset, keypoints = self.rnet(img_data)

        confid = confid.detach().cpu().numpy()
        offset = offset.detach().cpu().numpy()

        mask = confid[:, 0] > r_confid
        confid = confid[mask]
        offset = offset[mask]
        _boxes = boxes[mask]
        # print(_boxes.shape,confid.shape,offset.shape)

        # confid = confid.detach().numpy()
        # offset = offset.detach().numpy()
        res = np.hstack((offset, confid))

        _w = _boxes[:, 2] - _boxes[:, 0]
        _h = _boxes[:, 3] - _boxes[:, 1]

        res[:, 0] = res[:, 0] * _w + _boxes[:, 0]
        res[:, 1] = res[:, 1] * _h + _boxes[:, 1]
        res[:, 2] = res[:, 2] * _w + _boxes[:, 2]
        res[:, 3] = res[:, 3] * _h + _boxes[:, 3]
        # print(cal_nms(res, r_nms))
        # exit()
        print(len(res))
        res = nms(res, r_nms)
        print(len(res))
        return res

    def __o_detector(self, image, res_r):
        boxes = convert_to_square(res_r)
        # print(boxes)
        img_data = []
        for box in boxes:
            # img = image.crop(box)
            # img.show()
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            # img = image.crop((_x1, _y1, _x2, _y2))
            # img = img.resize((48, 48))
            img = image[_y1:_y2, _x1:_x2]
            img = cv2.resize(img, (48, 48))
            decode = transforms.ToTensor()
            img = decode(img)
            img_data.append(img)
        img_data = torch.stack(img_data).cuda()
        confid, offset, keypoints = self.onet(img_data)
        # print(confid.shape,offset.shape,keypoints.shape)

        confid = confid.detach().cpu().numpy()
        offset = offset.detach().cpu().numpy()
        keypoints = keypoints.detach().cpu().numpy()

        mask = confid[:, 0] > o_confid
        confid = confid[mask]
        offset = offset[mask]
        keypoints = keypoints[mask]
        _boxes = boxes[mask]
        # print(_boxes.shape)

        res = np.hstack((offset, confid, keypoints))
        # print(len(res))

        _w = _boxes[:, 2] - _boxes[:, 0]
        _h = _boxes[:, 3] - _boxes[:, 1]

        res[:, 0] = res[:, 0] * _w + _boxes[:, 0]
        res[:, 1] = res[:, 1] * _h + _boxes[:, 1]
        res[:, 2] = res[:, 2] * _w + _boxes[:, 2]
        res[:, 3] = res[:, 3] * _h + _boxes[:, 3]
        # keypoints
        res[:, 5] = res[:, 5] * _w + _boxes[:, 0]
        res[:, 6] = res[:, 6] * _h + _boxes[:, 1]
        res[:, 7] = res[:, 7] * _w + _boxes[:, 0]
        res[:, 8] = res[:, 8] * _h + _boxes[:, 1]
        res[:, 9] = res[:, 9] * _w + _boxes[:, 0]
        res[:, 10] = res[:, 10] * _h + _boxes[:, 1]
        res[:, 11] = res[:, 11] * _w + _boxes[:, 0]
        res[:, 12] = res[:, 12] * _h + _boxes[:, 1]
        res[:, 13] = res[:, 13] * _w + _boxes[:, 0]
        res[:, 14] = res[:, 14] * _h + _boxes[:, 1]

        # next_boxes = convert_square(res)
        # next_img_data = []
        # for box in next_boxes:
        #     _x1 = int(box[0])
        #     _y1 = int(box[1])
        #     _x2 = int(box[2])
        #     _y2 = int(box[3])
        #     img = image.crop((_x1, _y1, _x2, _y2))
        #     img = img.resize((48, 48))
        #     decode = transforms.ToTensor()
        #     img = decode(img)
        #     next_img_data.append(img)
        # next_img_data = torch.stack(next_img_data)
        # confid, offset, keypoints = self.onet(next_img_data)
        # confid = confid.detach().numpy()
        # res[:,4] = confid[:,0]
        # print(res)
        # exit()
        # return res
        return nms(res, o_nms, True)


if __name__ == '__main__':
    img_dir = r"D:\MTCNN\data\MTCNN-Pytorch\simple\data\test"
    for path in os.listdir(img_dir):
        img_path = os.path.join(img_dir, path)
        img = cv2.imread(img_path)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = Detector()
        boxes = detector.detector(img)
        if boxes.shape[0] == 0:
            print("No face found")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            confid = box[4]
            # print("confidence:", confid.item())

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        cv2.imshow("img", img)
        cv2.waitKey(0)
