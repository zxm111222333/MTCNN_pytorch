# 工具类: iou、nms、convert_to_square
import numpy


# 重叠率: 计算一个box与多个boxes的iou(交/并、交/最小框积)，并返回
# box形如[x1, y1, x2, y2, conf]，boxes形如[[x1, y1, x2, y2, conf], [x1, y1, x2, y2, conf], ...]
def iou(box, boxes, isMin=False):
    # 交集面积
    x1 = numpy.maximum(box[0], boxes[:, 0])
    y1 = numpy.maximum(box[1], boxes[:, 1])
    x2 = numpy.minimum(box[2], boxes[:, 2])
    y2 = numpy.minimum(box[3], boxes[:, 3])
    w = numpy.maximum(0, x2 - x1)
    h = numpy.maximum(0, y2 - y1)
    inter_areas = w * h

    # 计算框面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 计算iou
    if isMin:
        iou = numpy.true_divide(inter_areas, numpy.minimum(box_area, boxes_areas))
    else:
        # python3.0的/与numpy.true_divide(a, b)等价，都代表真正的除法运算符
        iou = inter_areas / (box_area + boxes_areas - inter_areas)
    return iou


# 非极大值抑制:
# 1.将根据置信度从大到小进行排序，得到输入列表
# 2.从list中移出置信度最大的框并添加到输出列表，并计算该框与其他框的所有iou。
# 3.根据阈值thresh进行筛选，排除iou>thresh的框。
# 4.对剩余的框重复2、3步骤，直到输入列表中没有元素。
# 5.最后返回outputs
# boxes形如: [[x1, y1, x2, y2, cls], [x1, y1, x2, y2, cls], ...]
def nms(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return numpy.array([])
    # 1.将根据置信度从大到小进行排序，得到输入框列表
    input_boxes = boxes[numpy.argsort(-boxes[:, 4])]

    # 定义输出列表
    output_boxes = []
    # while循环，直到长度<=1时停止
    while input_boxes.shape[0] > 1:
        # 2.从list中移出置信度最大的框并添加到输出列表，并计算该框与其他框的所有iou。
        first_box = input_boxes[0]  # 取第一个(置信度最大)框
        other_boxes = input_boxes[1:]  # 取剩余的框
        # 添加到输出列表中
        output_boxes.append(first_box)

        # 3.根据阈值thresh进行筛选，排除iou>thresh的框。
        # 因为循环判断条件是len>1，因此这里的other_boxes肯定是有元素的
        index = numpy.where(iou(first_box, other_boxes, isMin) <= thresh)  # 返回满足条件的索引
        input_boxes = other_boxes[index]  # 根据索引获取满足条件的框，并重新赋值给input_boxes继续循环
    if input_boxes.shape[0] > 0:
        output_boxes.append(input_boxes[0])

    # numpy.stack()组装为矩阵
    return numpy.stack(output_boxes)


# 将boxes框(不一定是正方形)转换为正方形框。这样可以保证resize(12, 12)后不变形。
def convert_to_square(boxes):
    if boxes.shape[0] == 0:
        return boxes
    square_boxes = boxes.copy()
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    # 取较大者作为正方形的边长
    side_len = numpy.maximum(w, h)
    # 坐标
    square_boxes[:, 0] = boxes[:, 0] + 0.5 * w - 0.5 * side_len
    square_boxes[:, 1] = boxes[:, 1] + 0.5 * h - 0.5 * side_len
    square_boxes[:, 2] = boxes[:, 0] + side_len
    square_boxes[:, 3] = boxes[:, 1] + side_len

    # square_boxes[:, 5] = boxes[:, 5] + 0.5 * w - 0.5 * side_len
    # square_boxes[:, 6] = boxes[:, 6] + 0.5 * h - 0.5 * side_len
    # square_boxes[:, 7] = boxes[:, 7] - 0.5 * w + 0.5 * side_len
    # square_boxes[:, 8] = boxes[:, 8] - 0.5 * h + 0.5 * side_len
    # square_boxes[:, 9] = boxes[:, 9] + 0.5 * w - 0.5 * side_len
    # square_boxes[:, 10] = boxes[:, 10] + 0.5 * h - 0.5 * side_len
    # square_boxes[:, 11] = boxes[:, 11] - 0.5 * w + 0.5 * side_len
    # square_boxes[:, 12] = boxes[:, 12] - 0.5 * h + 0.5 * side_len
    # square_boxes[:, 13] = boxes[:, 13] + 0.5 * w - 0.5 * side_len
    # square_boxes[:, 14] = boxes[:, 14] + 0.5 * h - 0.5 * side_len
    return square_boxes


if __name__ == '__main__':
    pass
    # # 1.测试iou()
    # box = numpy.array([1, 1, 11, 11])
    # # boxes = numpy.array([[1, 1, 10, 10], [11, 11, 20, 20]])
    # boxes = numpy.array([])
    # print(iou(box, boxes))

    # # 2.测试nms()
    # boxes = numpy.array([[1, 1, 11, 11, 0.7], [2, 2, 12, 12, 0.3], [11, 11, 22, 22, 0.8]])
    # output_boxes = nms(boxes)
    # print(output_boxes)

    # 3.测试conver_to_square()
    # boxes = numpy.array([1, 1, 11, 11], [2, 2, 8, 10], [3, 6, 8, 16])
