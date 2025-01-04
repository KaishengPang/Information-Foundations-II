# -*- coding: utf-8 -*-
# from __future__ import division
from __future__ import division,print_function
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.data
import skimage.util
import skimage.segmentation
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
import torchvision.datasets as datasets



def calculate_iou(box1, box2):
    # 计算交集的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # 计算交集的面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 计算并集的面积
    union = box1_area + box2_area - intersection
    # 计算IoU
    iou = intersection / union if union != 0 else 0
    return iou

# 定义一个函数来计算图像的纹理梯度
def compute_texture_gradient(image):
    # 初始化一个与图像形状相同的零数组
    texture_gradient = np.zeros(image.shape)
    # 对图像的每个通道进行操作
    for channel in (0, 1, 2):
        # 使用局部二值模式（LBP）计算每个通道的纹理梯度
        texture_gradient[:, :, channel] = skimage.feature.local_binary_pattern(
            image[:, :, channel], 8, 1.0)
    # 返回计算得到的纹理梯度
    return texture_gradient

# 定义一个函数来计算图像的颜色直方图
def calculate_color_histogram(image):
    # 设置直方图的箱子数
    BINS = 25
    # 初始化一个空的直方图数组
    histogram = np.array([])
    # 对图像的每个通道进行操作
    for channel in (0, 1, 2):
        # 计算每个通道的颜色直方图，并将其添加到总直方图中
        channel_histogram = np.histogram(image[:, channel], BINS, (0.0, 255.0))[0]
        histogram = np.concatenate([histogram, channel_histogram])
    # 归一化直方图
    histogram = histogram / len(image)
    # 返回计算得到的颜色直方图
    return histogram

# 定义一个函数来计算图像的纹理直方图
def calculate_texture_histogram(image):
    # 设置直方图的箱子数
    BINS = 10
    # 初始化一个空的直方图数组
    histogram = np.array([])
    # 对图像的每个通道进行操作
    for channel in (0, 1, 2):
        # 计算每个通道的纹理直方图，并将其添加到总直方图中
        channel_histogram = np.histogram(image[:, channel], BINS, (0.0, 1.0))[0]
        histogram = np.concatenate([histogram, channel_histogram])
    # 归一化直方图
    histogram = histogram / len(image)
    # 返回计算得到的纹理直方图
    return histogram

# 定义一个函数来计算两个区域的颜色相似性
def calculate_color_similarity(region1, region2):
    # 计算两个区域的颜色直方图的最小值之和
    return sum(min(a, b) for a, b in zip(region1["hist_c"], region2["hist_c"]))

# 定义一个函数来计算两个区域的纹理相似性
def calculate_texture_similarity(region1, region2):
    # 计算两个区域的纹理直方图的最小值之和
    return sum(min(a, b) for a, b in zip(region1["hist_t"], region2["hist_t"]))

# 定义一个函数来计算两个区域的大小相似性
def calculate_size_similarity(region1, region2, image_size):
    # 计算两个区域的大小与图像大小的差异
    return 1.0 - (region1["size"] + region2["size"]) / image_size

# 定义一个函数来计算两个区域的填充相似性
def calculate_fill_similarity(region1, region2, image_size):
    # 计算两个区域的边界框的大小
    Bx = max(region1["max_x"], region2["max_x"]) - min(region1["min_x"], region2["min_x"])
    By = max(region1["max_y"], region2["max_y"]) - min(region1["min_y"], region2["min_y"])
    BBsize = Bx * By
    # 计算两个区域的填充与图像大小的差异
    return 1.0 - (BBsize - region1["size"] - region2["size"]) / image_size

# 定义一个函数来找到图像中的区域
def find_regions(image):
    # 初始化一个空的区域字典
    regions = {}
    # 将图像从RGB空间转换为HSV空间
    hsv = skimage.color.rgb2hsv(image[:, :, :3])

    # 遍历图像的每个像素
    for y, row in enumerate(image):
        for x, (r, g, b, label) in enumerate(row):
            # 如果标签不在区域字典中，则添加到字典中
            if label not in regions:
                regions[label] = {"min_x": float('inf'), "min_y": float('inf'), 
                                  "max_x": 0, "max_y": 0, "labels": [label]}
            # 更新区域的边界
            regions[label]["min_x"] = min(regions[label]["min_x"], x)
            regions[label]["min_y"] = min(regions[label]["min_y"], y)
            regions[label]["max_x"] = max(regions[label]["max_x"], x)
            regions[label]["max_y"] = max(regions[label]["max_y"], y)

    # 计算图像的纹理梯度
    tex_grad = compute_texture_gradient(image)

    # 对每个区域进行操作
    for label, region in regions.items():
        # 获取属于当前区域的像素
        masked_pixels = hsv[:, :, :][image[:, :, 3] == label]
        # 计算区域的大小
        region["size"] = len(masked_pixels / 4)
        # 计算区域的颜色直方图
        region["hist_c"] = calculate_color_histogram(masked_pixels)
        # 计算区域的纹理直方图
        region["hist_t"] = calculate_texture_histogram(tex_grad[:, :][image[:, :, 3] == label])
    # 返回找到的区域
    return regions

# 定义一个函数来找到相邻的区域
def find_neighbouring_regions(regs):
    # 定义一个函数来判断两个区域是否重叠
    def overlaps(region_a, region_b):
        return any([region_a["min_x"] < region_b[coord] < region_a["max_x"] and 
                    region_a["min_y"] < region_b[coord2] < region_a["max_y"] 
                    for coord, coord2 in [("min_x", "min_y"), ("max_x", "max_y"), 
                                          ("min_x", "max_y"), ("max_x", "min_y")]])
    # 将区域字典转换为列表
    region_list = list(regs.items())
    # 找到所有重叠的区域对
    neighbours = [(a, b) for cur, a in enumerate(region_list[:-1]) for b in region_list[cur + 1:] if overlaps(a[1], b[1])]
    # 返回找到的相邻区域
    return neighbours

# 定义一个函数来给图像添加一个掩膜通道
def add_mask_channel(image, scale, sigma, min_size):
    # 使用Felzenszwalb方法进行图像分割，并得到掩膜
    mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(image), scale=scale, sigma=sigma, min_size=min_size)
    # 在图像的第三个维度上添加一个全零的通道
    image = np.append(image, np.zeros(image.shape[:2])[:, :, np.newaxis], axis=2)
    # 将掩膜赋值给新添加的通道
    image[:, :, 3] = mask
    # 返回添加了掩膜通道的图像
    return image

# 定义一个函数来计算两个区域的相似性
def calculate_similarity(region1, region2, image_size):
    # 计算颜色、纹理、大小和填充的相似性，并求和
    similarity = (calculate_color_similarity(region1, region2) + calculate_texture_similarity(region1, region2) +
                  calculate_size_similarity(region1, region2, image_size) + calculate_fill_similarity(region1, region2, image_size))
    # 返回相似性
    return similarity

# 定义一个函数来合并两个区域
def combine_regions(region1, region2):
    # 计算新区域的大小
    new_size = region1["size"] + region2["size"]
    # 创建一个新的区域，包含合并后的边界、大小、颜色直方图、纹理直方图和标签
    combined_region = {
        "min_x": min(region1["min_x"], region2["min_x"]),
        "min_y": min(region1["min_y"], region2["min_y"]),
        "max_x": max(region1["max_x"], region2["max_x"]),
        "max_y": max(region1["max_y"], region2["max_y"]),
        "size": new_size,
        "hist_c": (region1["hist_c"] * region1["size"] + region2["hist_c"] * region2["size"]) / new_size,
        "hist_t": (region1["hist_t"] * region1["size"] + region2["hist_t"] * region2["size"]) / new_size,
        "labels": region1["labels"] + region2["labels"]
    }
    # 返回合并后的区域
    return combined_region

# 定义一个函数来进行选择性搜索
def selective_search(image, scale=1.0, sigma=0.8, min_size=50):
    # 断言图像是三通道的
    assert image.shape[2] == 3, "输入非三通道图像"
    # 给图像添加一个掩膜通道
    img = add_mask_channel(image, scale, sigma, min_size)
    # 计算图像的大小
    img_size = img.shape[0] * img.shape[1]
    # 找到图像中的区域
    R = find_regions(img)
    # 找到相邻的区域
    neighbours = find_neighbouring_regions(R)
    # 计算每对相邻区域的相似性
    S = {(ai, bi): calculate_similarity(ar, br, img_size) for (ai, ar), (bi, br) in neighbours}

    # 当还有相似性大于0的区域对时，继续循环
    while S:
        # 找到相似性最大的区域对
        i, j = max(S.items(), key=lambda i: i[1])[0]
        # 创建一个新的标签
        t = max(R.keys()) + 1.0
        # 合并相似性最大的区域对，并添加到区域字典中
        R[t] = combine_regions(R[i], R[j])
        # 删除与合并区域有关的相似性记录
        keys_to_delete = [k for k, v in list(S.items()) if i in k or j in k]
        # 更新相似性字典
        S = {k: v for k, v in S.items() if k not in keys_to_delete}
        # 对于被删除的相似性记录，如果不是最大相似性的区域对，则重新计算相似性并添加到字典中
        for k in keys_to_delete:
            if k != (i, j):
                n = k[1] if k[0] in (i, j) else k[0]
                S[(t, n)] = calculate_similarity(R[t], R[n], img_size)

    # 创建一个区域列表，包含每个区域的矩形框、大小和标签
    regions = [{'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
                'size': r['size'], 'labels': r['labels']} for k, r in R.items()]
    # 返回添加了掩膜通道的图像和区域列表
    return img, regions

# 定义主函数
def main():
    # 下载并加载PASCAL VOC数据集
    voc_train = datasets.VOCDetection(root='./data', year='2012', image_set='train', download=True)
    # 获取第一张图片及其标注信息
    img, target = voc_train[0]
    # 提取边界框信息
    boxes = [[int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])] for obj in target['annotation']['object']]
    img = np.array(img)
    # 进行选择性搜索
    img_lbl, regions = selective_search(img, scale=500, sigma=0.5, min_size=100) # img, scale=1250, sigma=1.0, min_size=500
    # 计算原始候选区域的数量
    temp = {img_lbl[i, j, 3] for i in range(img_lbl.shape[0]) for j in range(img_lbl.shape[1])}
    print("原始候选区域:", len(temp))
    # 计算选择性搜索后的区域数量
    region_rect = {r['rect'] for r in regions if r['size'] >= 1000}
    print("selective_search区域", len(region_rect))

    # 设置IoU阈值
    iou_threshold = 0.5
    # 初始化一个空列表来存储筛选后的区域
    filtered_regions = []

    for rect in region_rect:
        # 将区域的坐标转换为边界框的格式
        predicted_box = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
        # 计算这个区域与所有真实边界框的IoU
        ious = [calculate_iou(predicted_box, gt_box) for gt_box in boxes]
        # 取最大的IoU作为这个区域的IoU
        max_iou = max(ious)
        print(f"预测边界框与真实边界框的交并比IoU {predicted_box} is {max_iou}")
        # 如果最大的IoU大于阈值，则将这个区域添加到筛选后的区域列表中
        if max_iou > iou_threshold:
            filtered_regions.append(rect)
    print(f"筛选后的区域数量：{len(filtered_regions)}")

    # 创建一个绘图窗口
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # 显示原始图像
    ax.imshow(img)
    # 在原始图像上绘制选择性搜索后的区域
    # for rect in region_rect:
    for rect in filtered_regions:
        ax.add_patch(mpatches.Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False, edgecolor='blue', linewidth=2))
    # 显示图像
    plt.show()

# 运行主函数
if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    # 运行主函数
    main()
    # 打印运行时间
    print("run time =", time.time() - start_time, "s")



    
