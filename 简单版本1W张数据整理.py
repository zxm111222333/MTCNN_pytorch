"""
数据：手动标记的边界框 + CelebA原始关键点
1.将手动标注的outputs文件转换成list_bbox_celeba.txt文件
2.合并list_bbox_celeba.txt + list_landmarks_celeba.txt = list_merge_celeba.txt
"""
import os
import json

if __name__ == '__main__':


    
    bbox_file = r"/result/simple\data\src\list_bbox_celeba.txt"
    landmarks_file = r"/result/simple\data\src\list_landmarks_celeba.txt"
    merge_file = r"/result/simple\data\src\list_merge_celeba.txt"
    bbox_list = []
    landmarks_list = []
    merge_list = []
    for line in open(bbox_file):
        bbox_list.append(line.strip().split())
    for line in open(landmarks_file):
        landmarks_list.append(line.strip().split())
    for i in range(len(bbox_list)):
        merge_item = bbox_list[i]
        if i == 1:
            merge_item.extend(landmarks_list[i])
        elif i > 1:
            merge_item.extend(landmarks_list[i][1:])
        merge_item_str = " ".join(merge_item)
        merge_list.append(merge_item_str)
    with open(merge_file, mode="w") as f:
        for i in range(len(merge_list)):
            print(f"写入第{i + 1}行数据")
            f.write(merge_list[i])
            f.write("\n")
        f.close()
