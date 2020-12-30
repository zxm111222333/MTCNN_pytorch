# 生成样本图片及对应的标签: 48*48
from gen_celeba_data import GenData

if __name__ == '__main__':
    # 原始图片路径和标签文件
    src_img_path1 = r"Q:\CelebA\Img\img_celeba.7z\img_celeba"
    src_anno_file1 = r"Q:\CelebA\Anno\list_merge_celeba.txt"
    # 生成的样本图片及标签的保存路径
    gen_save_path1 = r"D:\MTCNN\data\CelebA"

    for size in [48]:
        genData = GenData(src_img_path1, src_anno_file1, gen_save_path1, size)
        genData()
